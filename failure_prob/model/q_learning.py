import torch
import torch.nn as nn
from typing import Optional, Tuple,  List, Dict

from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad
from failure_prob.model.base import BaseModel
from failure_prob.conf import Config


def get_model(cfg, input_dim):
    """
    Factory function to create the appropriate Q-network model.
    
    Args:
        cfg: Configuration object
        input_dim: Input dimension
        
    Returns:
        Q-network model (either standard or categorical)
    """
    use_categorical = getattr(cfg.model, 'use_categorical', False)
    use_LSTM = getattr(cfg.model, 'use_LSTM', False)
    if use_categorical:
        return CategoricalGRUQNetwork(cfg, input_dim)
    elif use_LSTM:
        return LSTMQNetwork(cfg, input_dim)
    else:
        return GRUQNetwork(cfg, input_dim)

def compute_td_lambda_targets(
    next_q_values: torch.Tensor,
    success_labels: torch.Tensor,
    done_masks: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    max_horizon: int = 10,
) -> torch.Tensor:
    """
    Compute TD(λ) targets with episode-aware n-step returns in a vectorized manner.

    Args:
        next_q_values: Next step Q-values (B, T)
        success_labels: Episode success labels (B, T)
        done_masks: Mask where 0 indicates episode end (B, T)
        gamma: Discount factor
        lambda_: Lambda parameter for TD(λ)
        max_horizon: Maximum n-step horizon (default 10)
    
    Returns:
        TD(λ) targets (B, T - max_horizon + 1)
    """
    B, T = next_q_values.shape
    device = next_q_values.device
    dtype = next_q_values.dtype

    # Avoid clone: use torch.where to select between next_q_values and success_labels
    is_terminal = (done_masks == 0)
    values = torch.where(is_terminal, success_labels, next_q_values)

    # Precompute lambda powers (cache this if called repeatedly with same lambda_)
    lambdas = lambda_ ** torch.arange(max_horizon, device=device, dtype=dtype)

    # Unfold into sliding windows - reuse memory efficiently
    unfolded_values = values.unfold(1, max_horizon, 1)
    unfolded_done_masks = done_masks.unfold(1, max_horizon, 1)

    # Episode mask: shift and cumprod in one go using F.pad instead of torch.cat
    # Pad with 1s on the left, remove last column
    shifted_masks = torch.nn.functional.pad(unfolded_done_masks[:, :, :-1], (1, 0), value=1.0)
    episode_mask = torch.cumprod(shifted_masks, dim=2)

    # Combine operations: n_step_returns with mask applied
    n_step_returns = unfolded_values * episode_mask

    # Simplified weight calculation without building full matrix
    # For each actual horizon h, we need weights: [(1-λ)λ^0, (1-λ)λ^1, ..., (1-λ)λ^(h-2), λ^(h-1)]
    base_weights = (1 - lambda_) * lambdas  # Shape: (max_horizon,)

    # Calculate actual horizons
    actual_horizons = episode_mask.sum(dim=2, dtype=torch.long)  # (B, T_out)

    # Build weights directly without matrix indexing
    # Use broadcasting: expand base_weights and mask based on actual horizon
    T_out = actual_horizons.shape[1]
    weights = base_weights.unsqueeze(0).unsqueeze(0).expand(B, T_out, -1).clone()

    # Set the last valid weight in each row to λ^(h-1)
    # Create indices for batch and time dimensions
    batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, T_out)
    time_idx = torch.arange(T_out, device=device).view(1, -1).expand(B, -1)
    last_idx = (actual_horizons - 1).clamp(min=0, max=max_horizon-1)

    # Set the last weight using advanced indexing
    weights[batch_idx, time_idx, last_idx] = lambdas[last_idx]

    # Apply episode mask to weights
    final_weights = weights * episode_mask

    # Compute weighted sum - use sum instead of einsum for better performance
    lambda_returns = (n_step_returns * final_weights).sum(dim=2)

    return lambda_returns

def hl_gauss_projection(
        q_values: torch.Tensor,
        num_bins: int = 21,
        v_min: float = 0.0,
        v_max: float = 1.0,
        sigma_ratio: float = 0.75
) -> torch.Tensor:
    """
    Project Q-values to categorical distribution using HL-Gauss (Half-Logistic Gaussian).

    Based on the method from "Stop Regressing: Training Value Functions via Classification".
    Uses Gaussian distribution Y|S,A ~ N(μ=(TbQ)(S,A;θ⁻), σ²) and projects onto histogram bins.

    Args:
        q_values: Target Q-values μ = (TbQ)(S,A;θ⁻) from Bellman operator (B, T)
        num_bins: Number of bins m for categorical distribution
        v_min: Minimum value of support
        v_max: Maximum value of support
        sigma_ratio: σ/ς ratio - controls label smoothing (default 0.75 distributes mass to ~6 bins)

    Returns:
        Categorical distribution probabilities p_i(S,A;θ⁻) (B, T, num_bins)
    """
    # Precompute all constants in one go
    bin_width = (v_max - v_min) / num_bins
    inv_sigma = 1.0 / (sigma_ratio * bin_width)
    half_bin_scaled = 0.5 * bin_width * inv_sigma  # Pre-scaled half bin width
    
    # Create support and reshape in one operation
    z = torch.linspace(v_min, v_max, num_bins, device=q_values.device, dtype=q_values.dtype).view(1, 1, -1)
    
    # Fused operations: (z ± half_bin - mu) / sigma
    # Rearrange to: (z - mu) / sigma ± half_bin / sigma
    mu = q_values.unsqueeze(-1)
    z_normalized = (z - mu) * inv_sigma  # (B, T, num_bins) - vectorized standardization
    
    # Single ndtr call with fused addition/subtraction through broadcasting
    probs = torch.special.ndtr(z_normalized + half_bin_scaled) - torch.special.ndtr(z_normalized - half_bin_scaled)
    
    # Normalize: use multiplicative inverse to avoid division
    probs *= torch.reciprocal(probs.sum(dim=-1, keepdim=True).clamp_(min=1e-8))

    return probs

def _init_gruq_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')  # Better for ReLU
        # nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        # GRU has parameters like weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


class CategoricalGRUQNetwork(BaseModel):
    """
    GRU Q-network with categorical representation of the value distribution.
    
    Implements distributional RL where Q(s,a;θ) = E[Z(s,a;θ)], and Z is represented
    as a categorical distribution over discrete locations z_1, ..., z_m with probabilities
    p̂_i(s,a;θ) derived from logits l_i(s,a;θ) through softmax:
    
        p̂_i(s,a;θ) = exp(l_i(s,a;θ)) / Σ_j exp(l_j(s,a;θ))
        Z(s,a;θ) = Σ_i p̂_i(s,a;θ) · δ_z_i
        Q(s,a;θ) = Σ_i p̂_i(s,a;θ) · z_i
    """
    
    def __init__(self, cfg: Config, input_dim: int, is_target: bool = False):
        super().__init__(cfg, input_dim)
        self.use_actions = cfg.model.use_actions
        self.use_top_k_probs = getattr(cfg.model, 'use_top_k_probs', False)
        self.use_probs_features = getattr(cfg.model, 'use_probs_features', False)
        
        # Categorical distribution parameters
        self.num_atoms = getattr(cfg.model, 'num_atoms', 51)  # Number of bins m
        self.v_min = getattr(cfg.model, 'v_min', 0.0)  # Minimum value
        self.v_max = getattr(cfg.model, 'v_max', 1.0)  # Maximum value
        
        # Create support z_i for the categorical distribution
        self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.num_atoms))
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Calculate input dimension based on what inputs are used
        if self.use_top_k_probs:
            self.k_top_probs = getattr(cfg.model, 'k_top_probs', 10)
            prob_input_dim = cfg.model.prob_dim * self.k_top_probs
        elif self.use_probs_features:
            probs_feat_dim = getattr(cfg.model, 'probs_feat_dim', 4)
            prob_input_dim = cfg.model.prob_dim * probs_feat_dim
        else:
            prob_input_dim = cfg.model.prob_dim
        
        # Add action vector dimension if using actions
        input_dim = prob_input_dim + cfg.model.action_dim if self.use_actions else prob_input_dim
        
        self.target_network = None
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.model.gru_hidden),
            nn.LayerNorm(cfg.model.gru_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.model.gru_hidden, cfg.model.gru_hidden),
            nn.LayerNorm(cfg.model.gru_hidden),
            nn.ReLU(inplace=False),
        )
        
        self.gru_hidden = cfg.model.gru_hidden
        self.gru = nn.GRU(
            input_size=cfg.model.gru_hidden,
            hidden_size=cfg.model.gru_hidden,
            num_layers=cfg.model.num_gru_layers,
        )
        
        # Head outputs logits for each atom in the categorical distribution
        self.head = nn.Sequential(
            nn.Linear(cfg.model.gru_hidden, cfg.model.head_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.model.head_hidden, self.num_atoms),
        )
        
        self.apply(_init_gruq_weights)
        
        # Create target network if needed
        if cfg.model.loss in ["CategoricalTDLoss", "TDLoss", "TDLambdaLoss"] and not is_target:
            self.target_network = CategoricalGRUQNetwork(cfg, input_dim, is_target=True)
            self._copy_params_to_target()
            self.target_network.eval()
            self.steps = 0
    
    def _copy_params_to_target(self):
        """Copy parameters from main network to target network."""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
    
    def forward(
        self, 
        batch: dict[str, torch.Tensor], 
        hxs: Optional[torch.Tensor] = None,
        return_distribution: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            batch: Dictionary containing input tensors
            hxs: Hidden state for GRU
            return_distribution: If True, return probabilities; if False, return Q-values
            
        Returns:
            If return_distribution=False: Q-values (B, T, 1)
            If return_distribution=True: Distribution probabilities (B, T, num_atoms)
        """
        actions = batch["action_vectors"]
        not_done_masks = batch["done_masks"].squeeze(0) if "done_masks" in batch else None
        # if not_done_masks is not None and not_done_masks.dim() > 1:
        #     not_done_masks = not_done_masks[:, 0] # (T,)
        # Prepare input based on configuration
        if self.use_top_k_probs and "top_10_probs" in batch:
            top_k_probs = batch["top_10_probs"]
            B, T, action_dim, k = top_k_probs.shape
            probs = top_k_probs.reshape(B, T, -1)
        elif self.use_probs_features and "probs_features" in batch:
            probs_features = batch["probs_features"]
            B, T, D, _ = probs_features.shape
            probs = probs_features.reshape(B, T, -1)
        else:
            probs = batch["probabilities"]
            B, T, D = probs.shape
        
        if self.use_actions and actions is not None:
            x = torch.cat([probs, actions], dim=-1)
        else:
            x = probs
        
        x = self.input_proj(x)
        x = x.squeeze(0)
        
        if hxs is None:
            hxs = torch.zeros(1, x.size(1), dtype=x.dtype).to(x.device)
        
        if len(x.size()) > 2:  # Test mode
            hxs = torch.zeros(1, 1, x.size(2), dtype=x.dtype).to(x.device)
            outputs = []
            for i in range(x.size(0)):  # For every episode
                hxs = torch.zeros(1, x.size(2), dtype=x.dtype).to(x.device)
                ep_outputs = []
                for j in range(x.size(1)):  # For every timestep
                    out, hxs = self.gru(x[i:i+1, j:j+1, :].squeeze(0), hxs)
                    ep_outputs.append(out)
                outputs.append(torch.cat(ep_outputs, dim=0))
            x = torch.stack(outputs, dim=0)
        else:  # Train mode
            has_zeros = ((not_done_masks[1:] == 0).nonzero().squeeze().cpu())
            
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            
            has_zeros = [0] + has_zeros + [x.size(0)]
            
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                out, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * not_done_masks[start_idx]
                )
                outputs.append(out)
            
            x = torch.cat(outputs, dim=0)
        
        # Get logits for each atom
        logits = self.head(x)  # (T*B, num_atoms) or (B, T, num_atoms)
        
        # Apply softmax to get probabilities p̂_i(s,a;θ)
        probs = torch.softmax(logits, dim=-1)
        
        if return_distribution:
            return probs.unsqueeze(0) if len(x.size()) == 2 else probs
        
        # Compute Q-value as expected value: Q = Σ_i p̂_i · z_i
        q_values = (probs * self.support.view(1, -1)).sum(dim=-1, keepdim=True)
        
        if len(x.size()) > 2:  # Test mode
            scores = 1 - q_values
            return scores
        else:  # Train mode
            return q_values.unsqueeze(0)
    
    def project_distribution(
        self,
        next_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Project the value distribution using the Bellman operator.
        
        This implements the categorical projection algorithm for distributional RL.
        Given the next distribution and rewards, projects onto the current support.
        
        Args:
            next_probs: Next state distribution probabilities (B, T, num_atoms)
            rewards: Rewards (B, T)
            dones: Done masks (B, T), 1 = not done, 0 = done
            gamma: Discount factor
            
        Returns:
            Projected target distribution (B, T, num_atoms)
        """
        B, T = rewards.shape
        
        # Compute Tz = r + γz for each atom
        Tz = rewards.unsqueeze(-1) + gamma * dones.unsqueeze(-1) * self.support.view(1, 1, -1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        
        # Compute projection indices
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix edge cases
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability mass
        target_probs = torch.zeros_like(next_probs)
        offset = torch.linspace(0, (B * T - 1) * self.num_atoms, B * T, device=rewards.device).long()
        offset = offset.view(B, T, 1).expand(B, T, self.num_atoms)
        
        # Lower projection
        target_probs.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_probs * (u.float() - b)).view(-1)
        )
        
        # Upper projection
        target_probs.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_probs * (b - l.float())).view(-1)
        )
        
        return target_probs
    
    def forward_compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss using categorical cross-entropy between distributions.
        """
        device = batch["success_labels"].device
        use_categorical = self.cfg.model.loss == "CategoricalTDLoss"
        use_td = self.cfg.model.loss in ["TDLoss", "TDLambdaLoss", "CategoricalTDLoss"]
        
        # Prepare batches
        probs = batch["probabilities"]
        actions = batch["action_vectors"]
        done = batch["done_masks"]
        
        if use_td:
            max_horizon = self.cfg.model.td_horizon
            next_batch = {
                "probabilities": probs[:, 1:],
                "action_vectors": actions[:, 1:],
                "done_masks": done[:, 1:],
                "success_labels": batch["success_labels"][:, 1:],
            }
            if "top_10_probs" in batch and self.use_top_k_probs:
                next_batch["top_10_probs"] = batch["top_10_probs"][:, 1:]
            if self.use_probs_features and "probs_features" in batch:
                next_batch["probs_features"] = batch["probs_features"][:, 1:]
            
            current_batch = {
                "probabilities": probs[:, :-max_horizon],
                "action_vectors": actions[:, :-max_horizon],
                "done_masks": done[:, :-max_horizon],
                "success_labels": batch["success_labels"][:, :-max_horizon],
                "valid_masks": batch["valid_masks"][:, :-max_horizon],
            }
            if "top_10_probs" in batch and self.use_top_k_probs:
                current_batch["top_10_probs"] = batch["top_10_probs"][:, :-max_horizon]
            if self.use_probs_features and "probs_features" in batch:
                current_batch["probs_features"] = batch["probs_features"][:, :-max_horizon]
        else:
            current_batch = batch
        
        success_labels = current_batch["success_labels"]
        valid_masks = current_batch["valid_masks"]
        B, T = valid_masks.shape[:2]
        
        if use_categorical:
            # Get current distribution probabilities
            current_probs = self(current_batch, return_distribution=True).squeeze(0)  # (B, T, num_atoms)
            
            with torch.no_grad():
                # Get next Q-values from target network (not distributions)
                next_q_values = self.target_network(next_batch, return_distribution=False).squeeze(-1)  # (B, T)
                
                # Apply Bellman operator to get target Q-values
                not_done_masks = next_batch["done_masks"].squeeze(0)
                # For terminal states, use success labels as Q-values
                if not_done_masks is not None and (not_done_masks == 0).any():
                    # if not_done_masks.dim() > 1:
                    #     not_done_masks = not_done_masks[:, 0]
                    next_q_values[0, not_done_masks == 0] = next_batch["success_labels"][0, not_done_masks == 0]
                
                # Project target Q-values to categorical distribution using HL-Gauss
                # We use the configuration parameters for the projection
                sigma_ratio = getattr(self.cfg.model, 'sigma_ratio', 0.75)
                target_probs = hl_gauss_projection(
                    q_values=next_q_values,
                    num_bins=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    sigma_ratio=sigma_ratio
                )
            
            # Compute categorical cross-entropy loss
            losses = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1)  # (B, T)
            
        else:
            # Standard Q-learning
            q_values = self(current_batch).squeeze(-1)
            scores = 1 - q_values
            
            if use_td:
                with torch.no_grad():
                    next_q_values = self.target_network(next_batch).squeeze(-1)
                    not_done_masks = next_batch["done_masks"] if next_batch["done_masks"] is not None else torch.ones(B, T, device=device)
                    
                    if not_done_masks is not None and (not_done_masks == 0).any():
                        next_q_values[0, not_done_masks == 0] = next_batch["success_labels"][0, not_done_masks == 0]
                    next_scores = 1 - next_q_values
                
                losses = nn.functional.huber_loss(scores, next_scores, reduction='none')
            else:
                losses = nn.functional.mse_loss(scores, 1 - success_labels.unsqueeze(0), reduction='none')
        
        # Apply time weighting
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)
        time_weights = time_weights.to(losses)
        losses[success_labels == 0] *= time_weights[success_labels == 0]
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses.squeeze(0).unsqueeze(-1), valid_masks.squeeze(0).unsqueeze(-1), 
            success_labels.squeeze(0), weights, self.cfg.model.one_loss_per_seq,
        )
        
        if use_td:
            self.steps += 1
            if self.steps % self.cfg.model.target_update_freq == 0:
                self._copy_params_to_target()
        
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": None,
        }
        
        return monitor_loss, logs
                
class GRUQNetwork(BaseModel):
    """
    Larger GRU Q-network that accepts only a 7-dimensional probability vector as input (no time dimension).
    """
    # def __init__(
    #     self,
    #     prob_dim: int = 7,
    #     action_dim: int = 7,
    #     gru_hidden: int = 1024,
    #     head_hidden: int = 512,
    #     num_gru_layers: int = 1,
    #     use_actions: bool = False,
    #     device: str = "cuda:0"
    # ):
    def __init__(self, cfg: Config, input_dim: int, is_target: bool = False):
        super().__init__(cfg, input_dim)
        # self.device = device
        self.use_actions = cfg.model.use_actions
        self.use_top_k_probs = getattr(cfg.model, 'use_top_k_probs', False)
        self.use_probs_features = getattr(cfg.model, 'use_probs_features', False)
        
        # Calculate input dimension based on what inputs are used
        if self.use_top_k_probs:
            # Using top-k probabilities: (7, 10) -> flattened to 70
            # self.action_dim = cfg.model.prob_dim  # 7
            self.k_top_probs = getattr(cfg.model, 'k_top_probs', 10)  # 10
            prob_input_dim = cfg.model.prob_dim * self.k_top_probs  # 70
        elif self.use_probs_features:
            # Using probability features
            probs_feat_dim = getattr(cfg.model, 'probs_feat_dim', 4)
            prob_input_dim = cfg.model.prob_dim * probs_feat_dim  # 28
        else:
            # Using single probability per action: 7
            prob_input_dim = cfg.model.prob_dim
        
        # Add action vector dimension if using actions
        input_dim = prob_input_dim + cfg.model.action_dim if self.use_actions else prob_input_dim
        
        self.target_network = None

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.model.gru_hidden),
            nn.LayerNorm(cfg.model.gru_hidden),  # ADD THIS
            nn.ReLU(inplace=False),
            nn.Linear(cfg.model.gru_hidden, cfg.model.gru_hidden),
            nn.LayerNorm(cfg.model.gru_hidden),  # ADD THIS
            nn.ReLU(inplace=False),
        )
        
        self.gru_hidden = cfg.model.gru_hidden
        # self.gru = nn.LSTM(
        #     cfg.model.gru_hidden, 
        #     cfg.model.gru_hidden, 
        #     cfg.model.num_gru_layers,
        #     batch_first=True,
        #     dropout=cfg.model.dropout,
        # )
        self.gru = nn.GRU(
            input_size=cfg.model.gru_hidden,
            hidden_size=cfg.model.gru_hidden,
            num_layers=cfg.model.num_gru_layers,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.model.gru_hidden, cfg.model.head_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.model.head_hidden, 1),
            # nn.Softmax(dim=0),
        )
        self.apply(_init_gruq_weights)
        
        # Only create target network if this is not already a target network and loss is TDLoss or TDLambdaLoss
        if cfg.model.loss in ["TDLoss", "TDLambdaLoss"] and not is_target:
            self.target_network = GRUQNetwork(cfg, input_dim, is_target=True)
            self._copy_params_to_target()
            self.target_network.eval()
            self.steps = 0
    def _copy_params_to_target(self):
        """Copy parameters from main network to target network without using state_dict."""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(param.data)


    def forward(
        self, 
        batch: dict[str, torch.Tensor], hxs: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        # probs: (B, T, 7) or (B, T, 7, 10) depending on use_top_k_probs
        # actions: (B, T, 7) - optional

        # Concatenate actions if provided
        actions = batch["action_vectors"]
        not_done_masks = batch["done_masks"].squeeze(0) if "done_masks" in batch else None  # (B, T)
        # if not_done_masks is not None and not_done_masks.dim() > 1:
        #     not_done_masks = not_done_masks[:, 0] # (T,)
        # Use top-k probabilities if available, otherwise use regular probabilities
        if self.use_top_k_probs and "top_10_probs" in batch:
            top_k_probs = batch["top_10_probs"]  # (B, T, 7, 10)
            B, T = top_k_probs.shape[:2]
            # Flatten (B, T, 7, 10) -> (B, T, 70)
            probs = top_k_probs.reshape(B, T, -1)

        elif self.use_probs_features and "probs_features" in batch:
            probs_features = batch["probs_features"]  # (B, T, 28)
            B, T, D, _ = probs_features.shape
            probs = probs_features.reshape(B, T, -1)

        else:
            probs = batch["probabilities"]  # (B, T, 7)
            B, T, D = probs.shape
        
        if self.use_actions and actions is not None:
            x = torch.cat([probs, actions], dim=-1)
        else:
            x = probs

        x = self.input_proj(x)
        x = x.squeeze(0)

        if hxs is None:
            hxs = torch.zeros(1, x.size(1), dtype=x.dtype).to(x.device)
    
        if len(x.size()) > 2:  # test
            hxs = torch.zeros(1, 1, x.size(2),dtype=x.dtype).to(x.device)
            # run each episode separately and propagate its hxs
            # then run each timestep
            outputs = []
            for i in range(x.size(0)):# for every episode
                hxs = torch.zeros(1, x.size(2),dtype=x.dtype).to(x.device)
                ep_outputs = []
                for j in range(x.size(1)): # for every timestep
                    out, hxs = self.gru(x[i:i+1, j:j+1, :].squeeze(0), hxs)
                # out, hxs = self.gru(x[:, i:i+1, :], hxs)
                    ep_outputs.append(out)
                outputs.append(torch.cat(ep_outputs, dim=0))
            x = torch.stack(outputs, dim=0)
            logits = self.head(x)
            q = torch.sigmoid(logits)
            scores = 1 - q
            return scores  # (1, B)
            # x, hxs = self.gru(x)
            # logits = self.head(x)
            # q = torch.sigmoid(logits)
            # return q
        else:  # train
            # hxs = hxs.detach()
            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((not_done_masks[1:] == 0) \
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # Add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [x.size(0)]

            # hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                out, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * not_done_masks[start_idx])  # .unsqueeze(0) .view(1, -1, 1)
                outputs.append(out)

            # Assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

        logits = self.head(x)
        q = torch.sigmoid(logits)
        return q.unsqueeze(0)  # (1, B)
    
    
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        
        # Cache device and extract dimensions once
        device = batch["success_labels"].device
        next_batch = None
        use_td = self.cfg.model.loss in ["TDLoss", "TDLambdaLoss"]
        loss_type = self.cfg.model.loss
        # Prepare batch slicing for TD losses (create views, not copies)
        probs = batch["probabilities"]
        actions = batch["action_vectors"]
        done = batch["done_masks"]
        if use_td:
            # Create next_batch using sliced views
            max_horizon = self.cfg.model.td_horizon
            next_batch = {
                "probabilities": probs[:, 1:],
                "action_vectors": actions[:, 1:],
                "done_masks": done[:, 1:],
                "success_labels": batch["success_labels"][:, 1:],
            }
            if "top_10_probs" in batch and self.use_top_k_probs:
                top_k_probs = batch["top_10_probs"]
                next_batch["top_10_probs"] = top_k_probs[:, 1:]
            if self.use_probs_features and "probs_features" in batch:
                prob_features = batch["probs_features"]
                next_batch["probs_features"] = prob_features[:, 1:]

            current_batch = {
                "probabilities": probs[:, :-max_horizon],
                "action_vectors": actions[:, :-max_horizon],
                "done_masks": done[:, :-max_horizon],
                "success_labels": batch["success_labels"][:, :-max_horizon],
                "valid_masks": batch["valid_masks"][:, :-max_horizon],
            }
            if "top_10_probs" in batch and self.use_top_k_probs:
                current_batch["top_10_probs"] = batch["top_10_probs"][:, :-max_horizon]
            if self.use_probs_features and "probs_features" in batch:
                current_batch["probs_features"] = batch["probs_features"][:, :-max_horizon]
        else:
            current_batch = batch

        success_labels = current_batch["success_labels"]
        valid_masks = current_batch["valid_masks"]

        B, T = success_labels.shape[:2]
        
        # Forward pass
        q_values = self(current_batch)  # (B, T, 1)
        scores = (1 - q_values).squeeze(-1)  # (B, T)

        if loss_type in ["TDLoss", "TDLambdaLoss"]:
            # Get next Q-values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_batch).squeeze(-1)  # (B, T)
                
                if loss_type == "TDLoss":
                    # Simple TD(0) - just use next Q-values
                    not_done_masks = next_batch["done_masks"].squeeze(0) if next_batch["done_masks"] is not None else None
                    if not_done_masks is not None and (not_done_masks == 0).any():
                        # if not_done_masks.dim() > 1:
                        #     not_done_masks = not_done_masks[:, 0]
                        next_q_values[0, not_done_masks == 0] = next_batch["success_labels"][0, not_done_masks == 0]
                    next_scores = 1 - next_q_values
                else:
                    # TD(λ) with n-step returns
                    not_done_masks = next_batch["done_masks"] if next_batch["done_masks"] is not None else torch.ones(B, T, device=device)
                    # if not_done_masks is not None and not_done_masks.dim() > 1:
                    #     not_done_masks = not_done_masks[:, 0]
                    lambda_ = getattr(self.cfg.model, 'td_lambda', 0.95)

                    lambda_targets = compute_td_lambda_targets(
                        next_q_values=next_q_values,
                        success_labels=next_batch["success_labels"],
                        done_masks=not_done_masks,
                        gamma=1.0,
                        lambda_=lambda_,
                        max_horizon=max_horizon,
                    )
                    next_scores = 1 - lambda_targets

            ##########################################################
            # gauss_scores = hl_gauss_projection(q_values=scores)  # (B, T, num_bins)
            # gauss_next_scores = hl_gauss_projection(q_values=next_scores)  # (B, T, num_bins)
            # # # Compute cross-entropy loss between projected distributions
            # losses = -(gauss_next_scores * torch.log(gauss_scores + 1e-8)).sum(-1)
            # # losses = nn.functional.cross_entropy(gauss_scores.squeeze(0), gauss_next_scores.squeeze(0), reduction='none')
            # # losses = losses.unsqueeze(0)  # (B, T, 1)
            ##########################################################
            losses = nn.functional.huber_loss(scores, next_scores, reduction='none')  # (B, T)
            # losses = nn.functional.mse_loss(scores, next_scores, reduction='none')  # (B, T)
            ##########################################################
            # # Add BCE regularization from actual success/failure labels
            # bce_criterion = nn.BCELoss(reduction='none')
            # # Expand labels to match scores shape: failure is positive class (1 - success_labels)
            # labels_expanded = (1 - success_labels.float()).expand_as(scores)  # (B, T)
            # bce_losses = bce_criterion(scores, labels_expanded)  # (B, T)

            # # Combine TD loss with BCE regularization
            # lambda_bce = getattr(self.cfg.model, 'lambda_bce_reg', 0.001)  # Weight for BCE regularization
            # losses = td_losses + lambda_bce * bce_losses  # (B, T)

        elif loss_type == "MSELoss":
            losses = nn.functional.mse_loss(scores, 1 - success_labels, reduction='none')
            
        elif loss_type == "BCELoss":
            losses = nn.functional.binary_cross_entropy(scores, 1 - success_labels, reduction='none')
            
        elif loss_type == "CELoss":
            losses = nn.functional.cross_entropy(scores.unsqueeze(0), 1 - success_labels.unsqueeze(0), reduction='none')
        else:
            raise ValueError(f"Invalid loss function: {loss_type}")
        
        # Check for NaN
        if scores.isnan().any():
            raise ValueError("NaN detected in scores during loss computation.")        
        
        # Apply the time weights only on the failure samples
        # if valid_masks.dim() > 2:
        #     valid_masks = valid_masks[:, :, 0]  # (B, T)
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)
        time_weights = time_weights.to(scores)
        losses[success_labels == 0] *= time_weights[success_labels == 0]

        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses.squeeze(0).unsqueeze(-1), valid_masks.squeeze(0).unsqueeze(-1),
            success_labels.squeeze(0), weights, self.cfg.model.one_loss_per_seq,
        )
        
        if use_td:
            # Update the target network periodically
            self.steps += 1
            if self.steps % self.cfg.model.target_update_freq == 0:
                self._copy_params_to_target()

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": None,
        }

        return monitor_loss, logs


class LSTMQNetwork(GRUQNetwork):
    """
    LSTM Q-network that inherits from GRUQNetwork.
    Only overrides the recurrent layer to use LSTM instead of GRU.
    Uses LSTM for better long-term memory compared to GRU.
    """
    def __init__(self, cfg: Config, input_dim: int, is_target: bool = False):
        # Call parent __init__ but we'll replace the GRU with LSTM
        super().__init__(cfg, input_dim, is_target=True)  # Temporarily set is_target=True to skip target network creation
        
        # Replace GRU with LSTM
        self.lstm = nn.LSTM(
            input_size=cfg.model.gru_hidden,
            hidden_size=cfg.model.gru_hidden,
            num_layers=cfg.model.num_gru_layers,
            batch_first=True,  # Process batches with (B, T, D) format
        )
        self.gru_hidden = int(cfg.model.gru_hidden)
        delattr(self, 'gru')  # Remove the GRU layer
        
        # Create target network if needed (now that LSTM is set up)
        if cfg.model.loss in ["TDLoss", "TDLambdaLoss"] and not is_target:
            self.target_network = LSTMQNetwork(cfg, input_dim, is_target=True)
            self._copy_params_to_target()
            self.target_network.eval()
            self.steps = 0

    def forward(
        self, 
        batch: dict[str, torch.Tensor], 
        hxs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through LSTM network.
        Uses batch_first=True LSTM, so expects (B, T, D) input format.
        """
        actions = batch["action_vectors"]
        not_done_masks = batch["done_masks"].squeeze(0) if "done_masks" in batch else None
        is_training = not_done_masks is not None

        # Use top-k probabilities if available, otherwise use regular probabilities
        if self.use_top_k_probs and "top_10_probs" in batch:
            top_k_probs = batch["top_10_probs"]
            B, T, action_dim, k = top_k_probs.shape
            probs = top_k_probs.reshape(B, T, -1)
        elif self.use_probs_features and "probs_features" in batch:
            probs_features = batch["probs_features"]
            B, T, D, _ = probs_features.shape
            probs = probs_features.reshape(B, T, -1)
        else:
            probs = batch["probabilities"]
            B, T, D = probs.shape
        
        if self.use_actions and actions is not None:
            x = torch.cat([probs, actions], dim=-1)
        else:
            x = probs

        x = self.input_proj(x)  # (B, T, D)

        # LSTM with batch_first=True processes (B, T, D) directly
        out, _ = self.lstm(x)  # (B, T, hidden_size)

        logits = self.head(out)  # (B, T, 1)
        q = torch.sigmoid(logits)  # (B, T, 1)

        if is_training:
            # Training: return q values (success probability)
            # Note: The loss function expects q values, not scores
            return q  # (B, T, 1)
        else:
            # Test: return scores (1 - q) which is failure probability
            scores = 1 - q
            return scores  # (B, T, 1)
