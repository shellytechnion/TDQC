import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss
from .q_learning import compute_td_lambda_targets

from failure_prob.conf import Config


def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return IndepModel(cfg, input_dim)


class IndepModel(BaseModel):
    '''
    In this model, we are treating the model features at each timestep independently.
    Each feature is projected to a single scalar value and accumulated throughout rollout
    '''
    def __init__(self, cfg: Config, input_dim: int, is_target: bool = False):
        super().__init__(cfg, input_dim)
        
        self.total_input_dim = input_dim * cfg.model.n_history_steps
        self.hidden_dim = cfg.model.hidden_dim
        
        # Build up the model
        projector = []
        if cfg.model.n_layers == 1:
            projector.append(nn.Linear(self.total_input_dim, 1))
        else:
            projector.append(nn.Linear(self.total_input_dim, self.hidden_dim))
            projector.append(nn.ReLU())
            for _ in range(cfg.model.n_layers - 2):
                projector.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                projector.append(nn.ReLU())
            projector.append(nn.Linear(self.hidden_dim, 1))
            
        if cfg.model.final_act_layer == "sigmoid":
            projector.append(nn.Sigmoid())
        elif cfg.model.final_act_layer == "relu":
            projector.append(nn.ReLU())
        elif cfg.model.final_act_layer == "none":
            pass
        else:
            raise ValueError(f"Unknown final activation: {cfg.model.final_act_layer}")
            
        self.projector = nn.Sequential(*projector)
        
        # Initialize target network for TD losses
        self.target_network = None
        if cfg.model.loss in ["TDLoss", "TDLambdaLoss"] and not is_target:
            self.target_network = IndepModel(cfg, input_dim, is_target=True)
            BaseModel.__init__(self.target_network, cfg, input_dim)
            self.target_network.total_input_dim = self.total_input_dim
            self.target_network.hidden_dim = self.hidden_dim
            
            # Rebuild projector for target network
            target_projector = []
            if cfg.model.n_layers == 1:
                target_projector.append(nn.Linear(self.total_input_dim, 1))
            else:
                target_projector.append(nn.Linear(self.total_input_dim, self.hidden_dim))
                target_projector.append(nn.ReLU())
                for _ in range(cfg.model.n_layers - 2):
                    target_projector.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                    target_projector.append(nn.ReLU())
                target_projector.append(nn.Linear(self.hidden_dim, 1))
            
            if cfg.model.final_act_layer == "sigmoid":
                target_projector.append(nn.Sigmoid())
            elif cfg.model.final_act_layer == "relu":
                target_projector.append(nn.ReLU())
            elif cfg.model.final_act_layer == "none":
                pass
            
            self.target_network.projector = nn.Sequential(*target_projector)
            self._copy_params_to_target()
            self.target_network.eval()
            self.steps = 0
    
    def _copy_params_to_target(self):
        """Copy parameters from main network to target network without using state_dict."""
        if self.target_network is not None:
            for target_param, param in zip(self.target_network.parameters(), self.parameters()):
                target_param.data.copy_(param.data)

        
    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert x.shape[-1] == self.input_dim, f"Input dim mismatch: {x.shape[-1]} != {self.input_dim}"

        x = self.projector(x) # (batch_size, seq_len, 1)
        
        # assert not (self.cfg.model.cumsum and self.cfg.model.rmean), "Cannot use both cumsum and rmean at the same time"

        if self.cfg.model.cumsum or self.cfg.model.rmean:
            # Accumulate the scores over the time dimension
            x = torch.cumsum(x, dim=-2) # (batch_size, seq_len, 1)
            
            # rmean will overwrite the cumsum
            if self.cfg.model.rmean:
                x = x / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1) # (batch_size, seq_len, 1)

        return x
        
    
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = batch["success_labels"].device
        use_td = self.cfg.model.loss in ["TDLoss", "TDLambdaLoss"]
        loss_type = self.cfg.model.loss
        
        valid_masks = batch["valid_masks"]
        labels = batch["success_labels"]
        B, T, D = batch["features"].shape
        
        scores = self(batch)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        # Design the weights based on time
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks) # (B, T)
        time_weights = time_weights.to(scores) # (B, T)
        
        if use_td:
            # Compute TD loss using target network on same batch
            with torch.no_grad():
                target_q_values = self.target_network(batch).squeeze(-1)  # (B, T)
                
                # Terminal rewards: 1 for failure (success_labels==0), 0 for success (success_labels==1)
                terminal_rewards = (1 - labels).unsqueeze(-1).float()  # (B, 1)

                if loss_type == "TDLoss":
                    # Simple TD(0): next_q_values[t] = target_q_values[t+1] for t < T-1, and terminal_reward for t = T-1
                    # Concatenate target Q-values with terminal reward: (B, T+1)
                    target_q_values*=valid_masks
                    last_valid_idx = valid_masks.sum(dim=1).long() - 1
                    target_q_values[torch.arange(B, device=device), last_valid_idx] = 1 - labels
                    # Also set the next position if it's within bounds
                    next_idx = last_valid_idx + 1
                    within_bounds = next_idx < T
                    if within_bounds.any():
                        batch_indices = torch.arange(B, device=device)[within_bounds]
                        target_q_values[batch_indices, next_idx[within_bounds]] = (1 - labels)[within_bounds]
                    target_with_terminal = torch.cat([target_q_values, target_q_values[:, -1:]], dim=-1)  # (B, T+1)
                    # Slice from index 1 onwards to get next values: (B, T)
                    target_scores = target_with_terminal[:, 1:]  # (B, T)
                else:
                    # TD(λ) with n-step returns
                    max_horizon = getattr(self.cfg.model, 'td_horizon', 1)
                    lambda_ = getattr(self.cfg.model, 'td_lambda', 0.95)
                    
                    # Create next_q_values by shifting and padding with terminal reward
                    # next_q_values[t] = target_q_values[t+1] for t < T-1, terminal_reward for t = T-1
                    target_q_values[:, -1] = 1 - labels
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
            # labels_expanded = (1 - labels.unsqueeze(-1).float()).expand_as(scores)  # (B, T)
            # bce_losses = bce_criterion(1 - scores, labels_expanded)  # (B, T)
            # # Combine TD loss with BCE regularization
            # lambda_bce = getattr(self.cfg.model, 'lambda_bce_reg', 0.01)  # Weight for BCE regularization
            losses = td_losses #+ lambda_bce * bce_losses  # (B, T)
            losses[labels == 0] *= time_weights[labels == 0] # (B, T)
        else:
            if self.cfg.model.loss == "regular_BCE":
                success_labels_expanded = labels.unsqueeze(-1).expand(B, T).float()  # (B, T)
                
                # If using cumsum/rmean, scores can be very large
                if self.cfg.model.cumsum or self.cfg.model.rmean:
                    # Scores are accumulated, use MSE loss instead
                    losses = nn.functional.mse_loss(scores, success_labels_expanded, reduction='none')
                elif self.cfg.model.final_act_layer == "sigmoid":
                    # Scores are already probabilities in [0, 1]
                    losses = nn.functional.binary_cross_entropy(1 - scores, success_labels_expanded, reduction='none')
                else:
                    # Scores are raw outputs without cumsum, use BCE with logits
                    # Clamp to prevent overflow
                    scores_clamped = torch.clamp(scores, min=-50, max=50)
                    losses = nn.functional.binary_cross_entropy_with_logits(scores_clamped, success_labels_expanded, reduction='none')
                
                # Apply the time weights only on the failure samples
                # failure_mask = (success_labels_expanded == 0)  # (B, T)
                # losses = torch.where(failure_mask, losses * time_weights, losses)
            else:
                # Compute the loss as if each sequence is successful or failure, then aggregate back to (B, T)
                higher_thresh = self.cfg.model.threshold
                lower_thresh = 0
                seq_loss_success = torch.relu(scores - lower_thresh)  # (B, T)
                if self.cfg.model.use_threshold:
                    seq_loss_fail = time_weights * torch.relu(higher_thresh - scores)
                else:
                    seq_loss_fail = time_weights * (- scores)
                    
                losses = (labels == 1).float()[:, None] * seq_loss_success + \
                    (labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(losses, valid_masks, labels, weights)
        
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
        }
        
        return monitor_loss, logs