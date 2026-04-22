import warnings
from matplotlib import pyplot as plt
import numpy as np

from failure_prob.data.utils import Rollout

def compute_mean_std(scores, max_length = None):
    '''
    Compute the mean and std of a list of lists of varying lengths
    
    Args:
        scores: List of lists
        max_length: The maximum length of the lists
    '''
    if len(scores) == 0:
        return np.array([]), np.array([])
    
    if max_length is None:
        max_length = max([len(s) for s in scores])
        
    scores_paded = np.full((len(scores), max_length), np.nan)
    for i, scores in enumerate(scores):
        scores_paded[i, :len(scores)] = scores
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(scores_paded, axis=0)
        std = np.nanstd(scores_paded, axis=0)
    return mean, std

def compute_mean_std_by_quantile(scores, n_quantiles=100):
    '''
    Compute the mean and std of a list of lists by quantile (percentage of completion)
    
    Args:
        scores: List of lists of varying lengths
        n_quantiles: Number of quantile points to interpolate (default 100 for percentages)
    
    Returns:
        quantiles: Array of quantile values [0, 0.01, 0.02, ..., 1.0]
        mean: Mean score at each quantile
        std: Std of scores at each quantile
    '''
    if len(scores) == 0:
        return np.array([]), np.array([]), np.array([])
    
    quantiles = np.linspace(0, 1, n_quantiles)
    scores_at_quantiles = np.zeros((len(scores), n_quantiles))
    
    for i, score_seq in enumerate(scores):
        T = len(score_seq)
        if T == 0:
            scores_at_quantiles[i, :] = np.nan
            continue
            
        # For each quantile, find the corresponding timestep
        for j, q in enumerate(quantiles):
            timestep = int(q * (T - 1))  # Map quantile to timestep
            timestep = min(timestep, T - 1)  # Clamp to valid range
            scores_at_quantiles[i, j] = score_seq[timestep]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(scores_at_quantiles, axis=0)
        std = np.nanstd(scores_at_quantiles, axis=0)
    
    return quantiles, mean, std


def plot_curves(success_scores, fail_scores, ax, individual, use_quantiles=True, success_rollouts=None, fail_rollouts=None):
    """
    Plot score curves over time.
    
    Args:
        success_scores: List of score sequences for successful episodes
        fail_scores: List of score sequences for failed episodes
        ax: Matplotlib axis
        individual: If True, plot individual trajectories
        use_quantiles: If True, plot by percentage of completion; if False, use absolute timesteps
        success_rollouts: List of Rollout objects for successful episodes (optional, for metadata)
        fail_rollouts: List of Rollout objects for failed episodes (optional, for metadata)
    """
    if individual:
        for idx, scores in enumerate(success_scores):
            # Create label with metadata if rollout info available
            label = "Success"
            if success_rollouts is not None and idx < len(success_rollouts):
                rollout = success_rollouts[idx]
                label = f"Success_seed_{rollout.episode_idx}_task_{rollout.task_id}"

            if use_quantiles:
                quantiles = np.linspace(0, 1, len(scores))
                ax.plot(quantiles * 100, scores, color="green", alpha=0.5, label=label)
            else:
                ax.plot(scores, color="green", alpha=0.5, label=label)

        for idx, scores in enumerate(fail_scores):
            # Create label with metadata if rollout info available
            label = "Fail"
            if fail_rollouts is not None and idx < len(fail_rollouts):
                rollout = fail_rollouts[idx]
                label = f"Fail_seed_{rollout.episode_idx}_task_{rollout.task_id}"

            if use_quantiles:
                quantiles = np.linspace(0, 1, len(scores))
                ax.plot(quantiles * 100, scores, color="red", alpha=0.5, label=label)
            else:
                ax.plot(scores, color="red", alpha=0.5, label=label)
    else:
        if use_quantiles:
            # Plot by percentage of completion
            quantiles, success_mean, success_std = compute_mean_std_by_quantile(success_scores)
            x_axis = quantiles * 100  # Convert to percentage
            ax.plot(x_axis, success_mean, color="green", label="Success")
            ax.fill_between(x_axis, success_mean - success_std, success_mean + success_std, 
                          color="green", alpha=0.3)

            quantiles, fail_mean, fail_std = compute_mean_std_by_quantile(fail_scores)
            ax.plot(x_axis, fail_mean, color="red", label="Fail")
            ax.fill_between(x_axis, fail_mean - fail_std, fail_mean + fail_std, 
                          color="red", alpha=0.3)
            
            ax.set_xlabel("Episode Completion (%)")
        else:
            # Plot by absolute timesteps (original behavior)
            max_length = max([len(s) for s in success_scores + fail_scores])
            
            success_mean, success_std = compute_mean_std(success_scores, max_length)
            ax.plot(success_mean, color="green", label="Success")
            ax.fill_between(range(max_length), success_mean - success_std, success_mean + success_std, 
                          color="green", alpha=0.3)

            fail_mean, fail_std = compute_mean_std(fail_scores, max_length)
            ax.plot(fail_mean, color="red", label="Fail")
            ax.fill_between(range(max_length), fail_mean - fail_std, fail_mean + fail_std, 
                          color="red", alpha=0.3)
            
            ax.set_xlabel("Time Step")

def plot_scores_by_splits(
    scores_by_split_name: dict[str, list[np.ndarray]],
    rollouts_by_split_name: dict[str, list[Rollout]],
    individual=False
):
    n_axes = len(scores_by_split_name)
    fig, axes = plt.subplots(1, n_axes, figsize=(6*n_axes, 6))

    # Handle single subplot case
    if n_axes == 1:
        axes = [axes]

    for i, (split_name, scores) in enumerate(scores_by_split_name.items()):
        ax = axes[i]
        rollouts = rollouts_by_split_name[split_name]
        assert len(scores) == len(rollouts)

        # Separate scores and rollouts by success/failure
        success_scores = [scores[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 1]
        fail_scores = [scores[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 0]
        success_rollouts = [rollouts[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 1]
        fail_rollouts = [rollouts[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 0]

        # Pass rollout metadata to plot_curves
        plot_curves(success_scores, fail_scores, ax, individual,
                   success_rollouts=success_rollouts if individual else None,
                   fail_rollouts=fail_rollouts if individual else None)
        # ax.set_xlabel("Time Step")
        ax.set_title(split_name)
        
    return fig, axes


def plot_roc_curves(
    data: list,
    method_name: str
):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {method_name}")
    
    for fpr, tpr, name, roc_auc in data:
        plt.plot(fpr, tpr, label=f"{name}, AUC={roc_auc*100:.2f}")
        
    plt.legend()
    return fig


def plot_prc_curves(
    data: list,
    method_name: str
):
    fig = plt.figure()
    plt.plot([0, 1], [1, 0], linestyle="--", color="gray")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PRC Curve for {method_name}")
    
    for rec, pre, split, prc_auc in data:
        plt.plot(rec, pre, label=f"{split}, AUC={prc_auc*100:.2f}")
        
    plt.legend()
    return fig