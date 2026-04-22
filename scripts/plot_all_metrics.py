"""
Create unified plots for all metrics (Brier and ECE) on val_seen and val_unseen splits.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import pickle
from collections import defaultdict
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

# File paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO_ROOT)

# Color and style mappings
COLOR_GROUPS = {
    "mlp": "#2E86AB",
    "lstm": "#94C05B",
    "qlearning": "#820263",
}

DISPLAY_METHOD_ORDER = [
    "MLP on features (BCE Loss)",
    "MLP on features (TDQC Loss)",
    "RNN on features (BCE Loss)",
    "RNN on features (TDQC Loss)",
    "RNN on top-10 probabilities (BCE Loss)",
    "RNN on top-10 probabilities (TDQC Loss)",
]

LABEL_MAPPING = {
    # LIBERO methods
    "openvla-10-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "openvla-10-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "openvla-10-lstm-lstm": "RNN on features (BCE Loss)",
    "openvla-10-lstm-lstm_TD0": "RNN on features (TDQC Loss)", 
    "openvla-10-q_learning-top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "openvla-10-q_learning-top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",
    # "openvla-10-q_learning-top_k_probs_TD0_best_2seeds": "RNN on top-10 probabilities (TDQC Loss)",
    
    # WidowX methods
    "openvla-widowx-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "openvla-widowx-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "openvla-widowx-lstm-lstm": "RNN on features (BCE Loss)",
    "openvla-widowx-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "openvla-widowx-q_learning-top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    # "openvla-widowx-q_learning-top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",
    "openvla-widowx-q_learning-top_k_probs": "RNN on top-10 probabilities (TDQC Loss)",

    # Droid methods
    "pizero_fast_droid-0510-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "pizero_fast_droid-0510-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero_fast_droid-0510-lstm-lstm": "RNN on features (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",

    # Pi0 FAST LIBERO methods
    "pizero_fast-default-indep-mlp_BCE" : "MLP on features (BCE Loss)",
    "pizero_fast-default-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero_fast-default-lstm-lstm": "RNN on features (BCE Loss)",
    "pizero_fast-default-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "pizero_fast-default-lstm-lstm_top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "pizero_fast-default-lstm-lstm_top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",

    #Pi0 LIBERO methods
    "pizero-default-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "pizero-default-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero-default-lstm-lstm":"RNN on features (BCE Loss)",
    "pizero-default-lstm-lstm_TD0": "RNN on features (TDQC Loss)",

    # UniVLA methods
    "univla-default-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "univla-default-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "univla-default-lstm-lstm_BCE": "RNN on features (BCE Loss)",
    "univla-default-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "univla-default-lstm-lstm_top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "univla-default-lstm-lstm_top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",
}

EPISODE_FILE_PATTERNS = [
    re.compile(r"(?P<task>.+?)--ep(?P<episode>\d+)--succ(?P<success>[01])"),
    re.compile(r"(?P<task>.+?)_ep(?P<episode>\d+)_succ(?P<success>[01])"),
]


LIBERO_SEED_TASK_SPLITS = [
    {"seen": [2, 8, 4, 9, 1, 6, 7], "unseen": [3, 0, 5]},
    {"seen": [2, 9, 6, 4, 0, 3, 1], "unseen": [7, 8, 5]},
    {"seen": [4, 1, 5, 0, 7, 2, 3], "unseen": [6, 9, 8]},
    {"seen": [5, 4, 1, 2, 9, 6, 7], "unseen": [0, 3, 8]},
    {"seen": [3, 8, 4, 9, 2, 6, 0], "unseen": [1, 5, 7]},
    {"seen": [9, 5, 2, 4, 7, 1, 0], "unseen": [8, 6, 3]},
    {"seen": [8, 1, 7, 0, 6, 5, 2], "unseen": [4, 3, 9]},
    {"seen": [8, 5, 0, 2, 1, 9, 7], "unseen": [3, 6, 4]},
    {"seen": [8, 6, 9, 0, 2, 5, 7], "unseen": [1, 4, 3]},
    {"seen": [8, 4, 7, 2, 1, 9, 3], "unseen": [0, 6, 5]},
    {"seen": [8, 2, 5, 6, 3, 1, 0], "unseen": [7, 4, 9]},
    {"seen": [7, 8, 2, 6, 4, 5, 1], "unseen": [3, 0, 9]},
    {"seen": [5, 8, 7, 0, 4, 9, 3], "unseen": [2, 1, 6]},
    {"seen": [3, 5, 6, 1, 4, 7, 8], "unseen": [9, 0, 2]},
    {"seen": [3, 9, 0, 5, 4, 2, 1], "unseen": [7, 6, 8]},
    {"seen": [2, 6, 1, 3, 7, 0, 9], "unseen": [4, 5, 8]},
    {"seen": [6, 2, 0, 7, 8, 4, 3], "unseen": [1, 5, 9]},
    {"seen": [7, 2, 5, 3, 4, 0, 9], "unseen": [8, 6, 1]},
    {"seen": [7, 9, 0, 4, 2, 1, 6], "unseen": [5, 8, 3]},
    {"seen": [1, 7, 9, 6, 8, 4, 3], "unseen": [0, 2, 5]},
    {"seen": [7, 1, 8, 5, 0, 2, 6], "unseen": [9, 4, 3]},
]

WIDOWX_SEED_TASK_SPLITS = [
    {"seen": [6, 2, 1, 7, 3, 0], "unseen": [5, 4]},
    {"seen": [7, 2, 1, 6, 0, 4], "unseen": [3, 5]},
    {"seen": [4, 1, 6, 2, 3, 7], "unseen": [5, 0]},
    {"seen": [5, 7, 4, 6, 3, 1], "unseen": [0, 2]},
    {"seen": [4, 7, 3, 0, 1, 5], "unseen": [6, 2]},
    {"seen": [7, 2, 4, 1, 0, 5], "unseen": [6, 3]},
    {"seen": [0, 5, 6, 7, 4, 3], "unseen": [1, 2]},
    {"seen": [2, 5, 0, 6, 3, 1], "unseen": [4, 7]},
    {"seen": [7, 0, 2, 6, 5, 1], "unseen": [4, 3]},
    {"seen": [7, 1, 2, 3, 0, 5], "unseen": [4, 6]},
    {"seen": [2, 3, 6, 7, 0, 4], "unseen": [5, 1]},
    {"seen": [2, 6, 4, 5, 7, 3], "unseen": [0, 1]},
    {"seen": [4, 6, 0, 2, 1, 5], "unseen": [7, 3]},
    {"seen": [1, 4, 3, 5, 6, 7], "unseen": [0, 2]},
    {"seen": [5, 6, 7, 2, 1, 4], "unseen": [0, 3]},
    {"seen": [2, 3, 1, 6, 7, 4], "unseen": [5, 0]},
    {"seen": [2, 4, 0, 3, 5, 7], "unseen": [6, 1]},
    {"seen": [4, 3, 2, 5, 0, 6], "unseen": [1, 7]},
    {"seen": [4, 5, 6, 7, 1, 0], "unseen": [3, 2]},
    {"seen": [1, 4, 3, 0, 2, 7], "unseen": [6, 5]},
    {"seen": [5, 7, 0, 1, 6, 4], "unseen": [2, 3]},
]

LIBERO_BENCHMARKS = {
    "OpenVLA-LIBERO",
    r"$\pi_0$-LIBERO",
    r"$\pi_0$-FAST-LIBERO",
    "UniVLA-LIBERO",
}

DROID_BENCHMARK = r"$\pi_0$-FAST-Droid"
DROID_FIXED_SUCCESS_RATE = 0.5


def _get_seed_task_splits_for_benchmark(benchmark_name: str):
    if benchmark_name in LIBERO_BENCHMARKS:
        return LIBERO_SEED_TASK_SPLITS
    if benchmark_name == "OpenVLA-WidowX":
        return WIDOWX_SEED_TASK_SPLITS
    return None


def _parse_setup_env_file(setup_file: Path) -> dict:
    """Parse exported environment variables from setup_envs.bash."""
    env_map = {}
    if not setup_file.exists():
        return env_map

    export_re = re.compile(r"^\s*export\s+([A-Z0-9_]+)=['\"]?([^'\"]+)['\"]?\s*$")
    with open(setup_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = export_re.match(line.strip())
            if match:
                env_map[match.group(1)] = match.group(2)
    return env_map


def _resolve_rollout_root(var_name: str, setup_env_vars: dict):
    """Get rollout root from process env, falling back to setup_envs.bash."""
    value = os.environ.get(var_name) or setup_env_vars.get(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def _parse_episode_success_from_filename(filename: str):
    """Parse (task, episode, success) tuple from rollout filename."""
    for pattern in EPISODE_FILE_PATTERNS:
        match = pattern.search(filename)
        if match:
            return (
                match.group("task"),
                int(match.group("episode")),
                int(match.group("success")),
            )
    return None


def _build_task_name_to_id_map(rollout_root: Path) -> dict:
    """Build a map from parsed task name to numeric task_id using rollout metadata."""
    task_name_to_id = {}
    for pkl_path in rollout_root.rglob("*.pkl"):
        parsed = _parse_episode_success_from_filename(pkl_path.name)
        if not parsed:
            continue

        task_name, _episode_id, _success = parsed
        if task_name in task_name_to_id:
            continue

        try:
            with open(pkl_path, "rb") as f:
                rollout = pickle.load(f)
            task_id = rollout.get("task_id")
            if task_id is not None:
                task_name_to_id[task_name] = int(task_id)
        except Exception:
            continue

    return task_name_to_id


def _extract_task_index(task_name: str, task_name_to_id_map: dict):
    """Extract numeric task index from task name or metadata map."""
    match = re.search(r"task[_-]?(\d+)", task_name)
    if match:
        return int(match.group(1))

    mapped = task_name_to_id_map.get(task_name)
    if mapped is None:
        return None

    try:
        return int(mapped)
    except (TypeError, ValueError):
        return None


def _extract_episode_successes(rollout_root: Path, task_name_to_id_map: dict):
    """Extract unique episode successes from filenames under a rollout root."""
    episodes = {}
    conflict_count = 0

    for file_path in rollout_root.rglob("*succ*"):
        if not file_path.is_file():
            continue

        parsed = _parse_episode_success_from_filename(file_path.name)
        if not parsed:
            continue

        task_name, episode_id, success = parsed

        # Include parent path scope so identical task names in nested folders do not collide.
        rel_parent = file_path.parent.relative_to(rollout_root)
        parent_key = str(rel_parent) if str(rel_parent) != "." else ""
        scoped_task_id = f"{parent_key}::{task_name}" if parent_key else task_name
        episode_key = (scoped_task_id, episode_id)
        task_index = _extract_task_index(task_name, task_name_to_id_map)

        if episode_key in episodes and episodes[episode_key]["success"] != success:
            conflict_count += 1
            continue

        episodes[episode_key] = {
            "success": success,
            "task_index": task_index,
        }

    return episodes, conflict_count


def _compute_seeded_split_baseline(episodes: dict, seed_task_splits: list, split_name: str):
    """Compute Brier baseline by averaging per-seed split-specific Brier scores."""
    split_key = "seen" if split_name == "val_seen" else "unseen"

    seed_brier_scores = []
    seed_avg_task_success_rates = []

    for seed_split in seed_task_splits:
        split_task_ids = set(seed_split[split_key])

        per_task_successes = defaultdict(list)
        split_successes = []

        for episode in episodes.values():
            task_index = episode["task_index"]
            if task_index is None or task_index not in split_task_ids:
                continue

            success = episode["success"]
            per_task_successes[task_index].append(success)
            split_successes.append(success)

        if not per_task_successes or not split_successes:
            continue

        per_task_rates = [float(np.mean(v)) for v in per_task_successes.values()]
        avg_task_success_rate = float(np.mean(per_task_rates))
        brier_score = float(np.mean([(s - avg_task_success_rate) ** 2 for s in split_successes]))

        seed_avg_task_success_rates.append(avg_task_success_rate)
        seed_brier_scores.append(brier_score)

    if not seed_brier_scores:
        return None

    return {
        "avg_task_success_rate": float(np.mean(seed_avg_task_success_rates)),
        "brier_vs_avg_task_success_rate": float(np.mean(seed_brier_scores)),
        "n_seeds": len(seed_brier_scores),
    }


def _compute_fixed_sr_baseline(episodes: dict, fixed_success_rate: float):
    """Compute Brier baseline against a fixed success rate."""
    if not episodes:
        return None

    successes = [episode["success"] for episode in episodes.values()]
    if not successes:
        return None

    brier_score = float(np.mean([(s - fixed_success_rate) ** 2 for s in successes]))
    return {
        "avg_task_success_rate": float(fixed_success_rate),
        "brier_vs_avg_task_success_rate": brier_score,
        "n_seeds": 1,
    }


def compute_benchmark_rollout_baselines() -> dict:
    """Compute rollout-derived baseline metrics for each benchmark."""
    setup_env_vars = _parse_setup_env_file(REPO_ROOT / "setup_envs.bash")

    openvla_root = _resolve_rollout_root("TDQC_OPENVLA_ROLLOUT_ROOT", setup_env_vars)
    openvla_widowx_root = _resolve_rollout_root("TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT", setup_env_vars)
    openpi_root = _resolve_rollout_root("TDQC_OPENPI_ROLLOUT_ROOT", setup_env_vars)
    openpi_droid_root = _resolve_rollout_root("TDQC_OPENPI_DROID_ROLLOUT_ROOT", setup_env_vars)
    univla_root = _resolve_rollout_root("TDQC_UNIVLA_ROLLOUT_ROOT", setup_env_vars)

    benchmark_candidates = {
        "OpenVLA-LIBERO": [
            openvla_root / "single-foward/libero_10" if openvla_root else None,
            openvla_root / "single-foward/libero_10__" if openvla_root else None,
            openvla_root / "libero_10" if openvla_root else None,
        ],
        "OpenVLA-WidowX": [
            openvla_widowx_root / "openvla_widowx" if openvla_widowx_root else None,
            openvla_widowx_root if openvla_widowx_root and openvla_widowx_root.name == "openvla_widowx" else None,
        ],
        r"$\pi_0$-FAST-Droid": [
            openpi_droid_root / "rollouts_all" if openpi_droid_root else None,
            openpi_droid_root,
        ],
        r"$\pi_0$-LIBERO": [
            openpi_root / "pi0-libero_10/env_records" if openpi_root else None,
            openpi_root / "pi0-libero_10" if openpi_root else None,
        ],
        r"$\pi_0$-FAST-LIBERO": [
            openpi_root / "pi0fast-libero_10/env_records" if openpi_root else None,
            openpi_root / "pi0fast-libero_10" if openpi_root else None,
        ],
        "UniVLA-LIBERO": [
            univla_root / "libero_10/eval/env_records" if univla_root else None,
            univla_root / "libero_10/eval" if univla_root else None,
            univla_root / "libero_10_usefast0/eval/env_records" if univla_root else None,
            univla_root / "libero_10_usefast0/eval" if univla_root else None,
        ],
    }

    baseline_by_benchmark = {}

    for benchmark_name, candidates in benchmark_candidates.items():
        best_result = None
        for candidate in [c for c in candidates if c is not None]:
            if not candidate.exists():
                continue

            task_name_to_id_map = _build_task_name_to_id_map(candidate)
            episodes, conflicts = _extract_episode_successes(candidate, task_name_to_id_map)
            if not episodes:
                continue

            split_metrics = {}
            if benchmark_name == DROID_BENCHMARK:
                fixed_metrics = _compute_fixed_sr_baseline(episodes, DROID_FIXED_SUCCESS_RATE)
                if fixed_metrics is not None:
                    split_metrics["val_seen"] = fixed_metrics
                    split_metrics["val_unseen"] = fixed_metrics
            else:
                seed_task_splits = _get_seed_task_splits_for_benchmark(benchmark_name)
                if seed_task_splits is not None:
                    seen_metrics = _compute_seeded_split_baseline(episodes, seed_task_splits, "val_seen")
                    unseen_metrics = _compute_seeded_split_baseline(episodes, seed_task_splits, "val_unseen")
                    if seen_metrics is not None:
                        split_metrics["val_seen"] = seen_metrics
                    if unseen_metrics is not None:
                        split_metrics["val_unseen"] = unseen_metrics

            if not split_metrics:
                continue

            candidate_result = {
                "source_dir": candidate,
                "conflicts_skipped": conflicts,
                "n_episodes": len(episodes),
                "split_metrics": split_metrics,
            }
            if best_result is None or candidate_result["n_episodes"] > best_result["n_episodes"]:
                best_result = candidate_result

        if best_result is None:
            print(f"Warning: could not compute rollout baseline for {benchmark_name}")
            continue

        baseline_by_benchmark[benchmark_name] = best_result["split_metrics"]
        for split_name, split_metrics in best_result["split_metrics"].items():
            print(
                f"{benchmark_name} [{split_name}]: avg task SR = {split_metrics['avg_task_success_rate']:.4f}, "
                f"Brier(avg task SR) = {split_metrics['brier_vs_avg_task_success_rate']:.4f} "
                f"({split_metrics['n_seeds']} seeds, {best_result['n_episodes']} episodes) "
                f"from {best_result['source_dir']}"
            )
        if best_result["conflicts_skipped"] > 0:
            print(
                f"  Note: skipped {best_result['conflicts_skipped']} episode entries "
                f"with conflicting success labels"
            )

    return baseline_by_benchmark

def get_method_color(method_name: str) -> str:
    """Get color for a method."""
    # Determine method type (check q_learning/top_k_probs before lstm, since some methods contain both)
    if 'q_learning' in method_name.lower() or 'qlearning' in method_name.lower() or 'top_k_probs' in method_name.lower() or 'Top-10' in method_name.lower():
        color = COLOR_GROUPS['qlearning']
    elif 'mlp' in method_name.lower() or 'indep' in method_name.lower():
        color = COLOR_GROUPS['mlp']
    elif 'lstm' in method_name.lower():
        color = COLOR_GROUPS['lstm']
    else:
        color = '#999999'

    return color

def plot_benchmark_subplot(ax, csv_path, title, split, metric_col, rollout_baseline=None):
    """Plot a single benchmark subplot as grouped bar chart by time quantile."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping {title}")
        ax.text(0.5, 0.5, f"{title}\n(Data not available)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Filter for the specified split
    df_val = df[df['split'] == split].copy()
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No {split} data)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Filter for specific time quantiles only
    target_quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
    df_val = df_val[df_val['time_quantile'].isin(target_quantiles)].copy()
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No data for target quantiles)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Get labels and colors
    df_val['label'] = df_val['method'].map(LABEL_MAPPING)
    df_val = df_val.dropna(subset=['label'])  # Remove methods not in mapping
    df_val['color'] = df_val['method'].apply(get_method_color)
    df_val['is_td_loss'] = df_val['label'].str.contains('TDQC Loss', na=False)
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No mapped methods)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Get unique methods in order
    method_order = [
        label for label in DISPLAY_METHOD_ORDER
        if label in df_val['label'].unique()
    ]
    
    # Get time quantiles
    time_quantiles = sorted(df_val['time_quantile'].unique())
    n_methods = len(method_order)
    n_quantiles = len(time_quantiles)
    
    # Bar width and positions
    bar_width = 0.13
    x_pos = np.arange(n_quantiles)
    
    # Store handles for legend
    legend_handles = {}
    
    # Plot bars for each method
    for i, method_label in enumerate(method_order):
        method_data = df_val[df_val['label'] == method_label].sort_values('time_quantile')
        offset = (i - n_methods/2 + 0.5) * bar_width
        
        color = method_data.iloc[0]['color']
        is_td_loss = method_data.iloc[0]['is_td_loss']
        hatch = '///' if is_td_loss else None
        
        bars = ax.bar(x_pos + offset,
                     method_data[metric_col],
                     bar_width,
                     yerr=method_data[metric_col.replace('_mean', '_std')],
                     label=method_label,
                     color=color,
                     alpha=0.8,
                     capsize=3,
                     hatch=hatch,
                     edgecolor='white' if is_td_loss else None,
                     linewidth=1.5 if is_td_loss else 0,
                     error_kw={'linewidth': 1.0})
        
        legend_handles[method_label] = bars
    
    # Customize subplot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q:.1f}' for q in time_quantiles])
    ax.set_xlabel('Time Steps Quantile')
    
    # Set ylabel based on metric
    if 'brier' in metric_col:
        ylabel = 'Brier Score'
    elif 'ece' in metric_col:
        ylabel = 'ECE'
    else:
        ylabel = metric_col
    
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    baseline_brier = None
    baseline_sr = None
    if 'brier' in metric_col and rollout_baseline:
        baseline_brier = rollout_baseline.get('brier_vs_avg_task_success_rate')
        baseline_sr = rollout_baseline.get('avg_task_success_rate')

    if baseline_brier is not None:
        ax.axhline(
            y=baseline_brier,
            color='black',
            linestyle=':',
            linewidth=2.0,
            alpha=0.9,
        )
        label_text = f'Brier(avg SR) = {baseline_brier:.2f}'
        if baseline_sr is not None:
            label_text += f' (SR={baseline_sr:.2f})'
        ax.text(
            0.98,
            0.98,
            label_text,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=13,
            color='black',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
        )
    
    # Set y-axis limits
    max_val = (df_val[metric_col] + df_val[metric_col.replace('_mean', '_std')]).max()
    if baseline_brier is not None:
        max_val = max(max_val, baseline_brier)
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.15)
    
    return legend_handles

def create_unified_plot(metric_type, split, output_filename, include_pi0_fast=False, benchmark_rollout_baselines=None):
    """Create a unified plot for a specific metric and split."""
    
    # Determine the metric column name
    if metric_type == 'brier':
        metric_col = 'brier_mean'
        metric_name = 'Brier Score'
    else:  # ece
        metric_col = 'ece_mean'
        metric_name = 'ECE'
    
    # File paths for this metric and split
    LIBERO_FILE = os.path.join(DATA_DIR, f"plots_libero/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    WIDOWX_FILE = os.path.join(DATA_DIR, f"plots_widowx/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    DROID_FILE = os.path.join(DATA_DIR, f"plots_droid/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    PI0_FILE = os.path.join(DATA_DIR, f"plots_pi0_libero/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    PI0_FAST_FILE = os.path.join(DATA_DIR, f"plots_pi0_fast_libero/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    UNIVLA_FILE = os.path.join(DATA_DIR, f"plots_univla/calibration_curves_model_{metric_type}_{split}_table_data.csv")
    
    # Determine layout based on whether to include pi0_fast
    if include_pi0_fast:
        fig, axes = plt.subplots(2, 3, figsize=(24, 8))
        axes = axes.flatten()
        benchmarks = [
            (LIBERO_FILE, "OpenVLA-LIBERO", axes[0]),
            (WIDOWX_FILE, "OpenVLA-WidowX", axes[1]),
            (DROID_FILE, r"$\pi_0$-FAST-Droid", axes[2]),
            (PI0_FILE, r"$\pi_0$-LIBERO", axes[3]),
            (PI0_FAST_FILE, r"$\pi_0$-FAST-LIBERO", axes[4]),
            (UNIVLA_FILE, "UniVLA-LIBERO", axes[5])
        ]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(24, 8))
        axes = axes.flatten()
        benchmarks = [
            (LIBERO_FILE, "OpenVLA-LIBERO", axes[0]),
            (WIDOWX_FILE, "OpenVLA-WidowX", axes[1]),
            (DROID_FILE, r"$\pi_0$-FAST-Droid", axes[2]),
            (PI0_FILE, r"$\pi_0$-LIBERO", axes[3]),
        ]
    
    all_handles = {}

    # Output directory for individual benchmark plots
    individual_dir = os.path.join(DATA_DIR, "plots_per_benchmark")
    os.makedirs(individual_dir, exist_ok=True)

    for csv_path, title, ax in benchmarks:
        rollout_baseline = None
        if metric_type == 'brier' and benchmark_rollout_baselines:
            benchmark_split_baselines = benchmark_rollout_baselines.get(title, {})
            rollout_baseline = benchmark_split_baselines.get(split)

        handles = plot_benchmark_subplot(
            ax,
            csv_path,
            title,
            split,
            metric_col,
            rollout_baseline=rollout_baseline,
        )
        all_handles.update(handles)

        # Save individual benchmark figure
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ind_handles = plot_benchmark_subplot(
            ax_ind, csv_path, title, split, metric_col,
            rollout_baseline=rollout_baseline,
        )
        # Add legend to individual plot
        ind_ordered_handles = []
        ind_ordered_labels = []
        for label in DISPLAY_METHOD_ORDER:
            if label in ind_handles:
                ind_ordered_handles.append(ind_handles[label])
                ind_ordered_labels.append(label)
        if ind_ordered_handles:
            fig_ind.legend(ind_ordered_handles, ind_ordered_labels,
                          loc='lower center', bbox_to_anchor=(0.5, -0.1),
                          ncol=2, fontsize=11, frameon=True)
        fig_ind.tight_layout(rect=[0, 0.05, 1, 1])
        safe_title = title.replace("$", "").replace("\\", "").replace(" ", "_").replace("_0$", "").replace("{", "").replace("}", "")
        ind_path = os.path.join(individual_dir, f"{safe_title}_{metric_type}_{split}.png")
        fig_ind.savefig(ind_path, dpi=300, bbox_inches='tight')
        print(f"  Saved individual plot: {ind_path}")
        plt.close(fig_ind)

    axes[5].set_visible(False)
    
    # Create single legend for all subplots
    ordered_handles = []
    ordered_labels = []
    for label in DISPLAY_METHOD_ORDER:
        if label in all_handles:
            ordered_handles.append(all_handles[label])
            ordered_labels.append(label)
    
    # Add legend below the subplots
    legend_y     = -0.05
    tight_bottom =  0.03 
    fig.legend(ordered_handles, ordered_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, legend_y),
              ncol=3, 
              fontsize=14,
              frameon=True)
    
    plt.tight_layout(rect=[0, tight_bottom, 1, 1])
    
    # Save figure
    output_path = os.path.join(DATA_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} {split} plot to: {output_path}")
    plt.close()

def main():
    """Create all four plots."""
    import argparse
    parser = argparse.ArgumentParser(description='Create unified metric plots')
    parser.add_argument('--include-pi0-fast', action='store_true', 
                       help='Include Pi0-FAST-LIBERO as 5th benchmark')
    args = parser.parse_args()

    benchmark_rollout_baselines = compute_benchmark_rollout_baselines()
    
    # Brier val_unseen (already exists, but recreate for consistency)
    pi0fast_eddition = "_with_pi0fast" if args.include_pi0_fast else ""
    create_unified_plot(
        'brier',
        'val_unseen',
        f'unified_brier_val_unseen{pi0fast_eddition}.png',
        args.include_pi0_fast,
        benchmark_rollout_baselines=benchmark_rollout_baselines,
    )
    
    # Brier val_seen
    create_unified_plot(
        'brier',
        'val_seen',
        f'unified_brier_val_seen{pi0fast_eddition}.png',
        args.include_pi0_fast,
        benchmark_rollout_baselines=benchmark_rollout_baselines,
    )
    
    # ECE val_unseen
    create_unified_plot(
        'ece',
        'val_unseen',
        f'unified_ece_val_unseen{pi0fast_eddition}.png',
        args.include_pi0_fast,
    )
    
    # ECE val_seen
    create_unified_plot(
        'ece',
        'val_seen',
        f'unified_ece_val_seen{pi0fast_eddition}.png',
        args.include_pi0_fast,
    )

if __name__ == '__main__':
    main()
