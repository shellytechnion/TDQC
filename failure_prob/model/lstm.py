import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad
from .q_learning import compute_td_lambda_targets

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return LstmModel(cfg, input_dim)


class LstmModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int, is_target: bool = False):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        
        # Add input projection when input is not hidden_states (probs / top_k_probs)
        _input_type = getattr(cfg.model, 'input_type', 'features')
        if _input_type != 'features':
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=False),
            )
            lstm_input_dim = self.hidden_dim
        else:
            self.input_proj = None
            lstm_input_dim = input_dim
        
        use_gru = getattr(cfg.model, 'use_gru', False)
        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.lstm = rnn_cls(
            lstm_input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            dropout=cfg.model.dropout,
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.n_history_steps = cfg.model.n_history_steps

        self._scale_weights(self.cfg.model.init_weight_scale)
        
        # Initialize target network for TD losses
        self.target_network = None
        if cfg.model.loss in ["TDLoss", "TDLambdaLoss"] and not is_target:
            self.target_network = LstmModel(cfg, input_dim, is_target=True)
            self._copy_params_to_target()
            self.target_network.eval()
            self.steps = 0
    
    def _copy_params_to_target(self):
        """Copy parameters from main network to target network without using state_dict."""
        if self.target_network is not None:
            for target_param, param in zip(self.target_network.parameters(), self.parameters()):
                target_param.data.copy_(param.data)
    

    def _select_input(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the input tensor based on cfg.model.input_type.
        Always normalises the last dim to self.input_dim via zero-padding or trimming,
        because pad_rollout_batch pads each batch independently to its own max token
        count, so different batches can produce different last-dim sizes.
        """
        input_type = getattr(self.cfg.model, 'input_type', 'features')
        if input_type == 'images' and 'images' in batch and batch['images'] is not None:
            return batch['images'].float()   # (B, T, embed_dim)
        if input_type == 'probs' and 'probabilities' in batch and batch['probabilities'] is not None:
            x = batch['probabilities'].float()          # (B, T, n_tokens)
        elif input_type == 'top_k_probs' and 'top_10_probs' in batch and batch['top_10_probs'] is not None:
            t = batch['top_10_probs'].float()           # (B, T, n_tokens, K)
            B, T = t.shape[:2]
            x = t.reshape(B, T, -1)                     # (B, T, n_tokens*K)
        else:
            return batch['features']

        # Normalise to self.input_dim
        current_dim = x.shape[-1]
        if current_dim < self.input_dim:
            pad = x.new_zeros(*x.shape[:-1], self.input_dim - current_dim)
            x = torch.cat([x, pad], dim=-1)
        elif current_dim > self.input_dim:
            x = x[..., :self.input_dim]
        return x

    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self._select_input(batch)
        B, T, D = x.shape
        n = self.n_history_steps

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"

        # Project input if using probs/top_k_probs
        if self.input_proj is not None:
            x = self.input_proj(x)  # (B, T, hidden_dim)

        if n < 0:
            out, _ = self.lstm(x)  # (B, T, hidden_dim)
        else:
            # Prepare sliding windows: for each timestep t, extract [t-n, ..., t-1]
            x_padded = torch.nn.functional.pad(x, (0, 0, n, 0), mode="constant", value=0)  # (B, T+n, D)

            x_windows = []
            for t in range(T):
                x_window = x_padded[:, t:t+n, :]  # (B, n, D)
                x_windows.append(x_window)

            x_seq = torch.stack(x_windows, dim=1)  # (B, T, n, D)
            x_seq = x_seq.reshape(B * T, n, D)     # (B*T, n, D)

            out, _ = self.lstm(x_seq)              # (B*T, n, hidden_dim)
            out = out[:, -1, :]                    # (B*T, hidden_dim)
            out = out.view(B, T, -1)               # (B, T, hidden_dim)

        out = self.dropout(out)                 # (B, T, hidden_dim)
        p_seq = torch.sigmoid(self.fc(out))    # (B, T, 1)

        if self.cfg.model.cumsum:
            # p_seq = p_seq.cumsum(dim=1)
            p_seq = cumsum_stopgrad(p_seq, dim=1)  # (B, T, 1)
            if self.cfg.model.rmean:
                normalizer = p_seq.new_ones(p_seq.shape).cumsum(dim=1)
                p_seq = p_seq / normalizer

        return p_seq


    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = batch["success_labels"].device
        use_td = self.cfg.model.loss in ["TDLoss", "TDLambdaLoss"]
        loss_type = self.cfg.model.loss
        
        _input_type = getattr(self.cfg.model, 'input_type', 'features')
        if _input_type == 'images' and 'images_valid_masks' in batch:
            valid_masks = batch["images_valid_masks"]
        else:
            valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]
        B = batch["features"].shape[0]

        scores = self(batch)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        T = scores.shape[1]
        
        # Design the weights based on time
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)  # (B, T)
        time_weights = time_weights.to(scores) # (B, T)
        
        if use_td:
            # Compute TD loss using target network on same batch
            with torch.no_grad():
                target_q_values = self.target_network(batch).squeeze(-1)  # (B, T)
                
                # Terminal rewards: 1 for failure (success_labels==0), 0 for success (success_labels==1)
                terminal_rewards = (1 - success_labels).unsqueeze(-1).float()  # (B, 1)

                if loss_type == "TDLoss":
                    # Simple TD(0): next_q_values[t] = target_q_values[t+1] for t < T-1, and terminal_reward for t = T-1
                    # Concatenate target Q-values with terminal reward: (B, T+1)
                    target_q_values*=valid_masks
                    last_valid_idx = valid_masks.sum(dim=1).long() - 1
                    target_q_values[torch.arange(B, device=device), last_valid_idx] = 1 - success_labels
                    # Also set the next position if it's within bounds
                    next_idx = last_valid_idx + 1
                    within_bounds = next_idx < T
                    if within_bounds.any():
                        batch_indices = torch.arange(B, device=device)[within_bounds]
                        target_q_values[batch_indices, next_idx[within_bounds]] = (1 - success_labels)[within_bounds]
                    target_with_terminal = torch.cat([target_q_values, target_q_values[:, -1:]], dim=-1)  # (B, T+1)
                    # Slice from index 1 onwards to get next values: (B, T)
                    target_scores = target_with_terminal[:, 1:]  # (B, T)
                else:
                    # TD(λ) with n-step returns
                    max_horizon = getattr(self.cfg.model, 'td_horizon', 1)
                    lambda_ = getattr(self.cfg.model, 'td_lambda', 0.95)
                    
                    # Create next_q_values by shifting and padding with terminal reward
                    # next_q_values[t] = target_q_values[t+1] for t < T-1, terminal_reward for t = T-1
                    target_q_values[:, -1] = 1 - success_labels
                    target_with_terminal = target_q_values[:, 1:]
                    for _ in range(max_horizon):
                        target_with_terminal = torch.cat([target_with_terminal, terminal_rewards], dim=-1)  # (B, T+1)
                    next_q_values = target_with_terminal  # (B, T)

                    # Create done_masks: 1 where episode is done (only at last valid timestep)
                    # We mark the last valid timestep as done
                    done_masks = torch.zeros(B, T + max_horizon - 1, device=device)

                    # Put 0 in the last valid timestep of valid_masks
                    last_valid_idx = valid_masks.sum(dim=1).long() - 1  # (B,) - index of last valid timestep
                    valid_masks[torch.arange(B, device=device), last_valid_idx] = 0

                    done_masks[:, :-max_horizon] = valid_masks[:,1:]  # Shifted valid masks

                    # Terminal rewards for all timesteps (broadcast from success_labels)
                    rewards = terminal_rewards.expand(B, T + max_horizon - 1)  # (B, T)

                    lambda_targets = compute_td_lambda_targets(
                        next_q_values=next_q_values,
                        success_labels=rewards,
                        done_masks=done_masks,
                        gamma=1.0,
                        lambda_=lambda_,
                        max_horizon=max_horizon,
                    )
                    target_scores = lambda_targets
                    valid_masks = batch["valid_masks"]

            # Compute TD loss (Huber loss)
            td_losses = nn.functional.mse_loss(scores, target_scores, reduction='none')  # (B, T)
            
            # Add BCE regularization from actual success/failure labels
            # bce_criterion = nn.BCELoss(reduction='none')
            # # Expand labels to match scores shape: failure is positive class (1 - success_labels)
            # labels_expanded = (1 - success_labels.unsqueeze(-1).float()).expand_as(scores)  # (B, T)
            # bce_losses = bce_criterion(scores, labels_expanded)  # (B, T)
            # # Combine TD loss with BCE regularization
            # lambda_bce = getattr(self.cfg.model, 'lambda_bce_reg', 0.001)  # Weight for BCE regularization
            losses = td_losses # + lambda_bce * bce_losses  # (B, T)
            losses[success_labels == 0] *= time_weights[success_labels == 0] # (B, T)
            
        elif self.cfg.model.cumsum:
            # Compute the loss as if each sequence is successful or failure, then aggregate back to (B, T)
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)  # (B, T)
            seq_loss_fail = time_weights * (- scores)
                
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + \
                (success_labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
        else:
            # Compute BCE loss on scores at all timesteps
            criterion = nn.BCELoss(reduction="none")
            # Failure is the positive class
            if scores.isnan().any():
                import pdb; pdb.set_trace()
            losses = criterion(scores, 1 - success_labels.unsqueeze(-1).expand_as(scores)) # (B, T)
            
            # Apply the time weights only on the failure samples
            losses[success_labels == 0] *= time_weights[success_labels == 0] # (B, T)
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses, valid_masks, success_labels, weights,
            self.cfg.model.one_loss_per_seq,
        )

        # Now that we want to do classification based on the max scores before termination
        # Therefore add hard nagative mining loss
        hard_neg_loss = torch.tensor(0.0).to(scores)
        if self.cfg.model.lambda_hard_heg > 0:
            # Note that in success_labels==0 means failure
            hard_neg_loss = hard_negative_loss(
                scores, 1-success_labels, valid_masks, 
                self.cfg.model.hard_neg_margin, 
                self.cfg.model.hard_neg_beta
            )
            hard_neg_loss = self.cfg.model.lambda_hard_heg * hard_neg_loss
        
        monitor_loss += hard_neg_loss
        
        # Update target network for TD losses
        if use_td:
            self.steps += 1
            if self.steps % self.cfg.model.target_update_freq == 0:
                self._copy_params_to_target()

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
        }
        
        return monitor_loss, logs