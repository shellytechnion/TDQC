#!/usr/bin/env python3
"""
Extract quantile ROC-AUC metrics from wandb and plot them.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Initialize API
api = wandb.Api()

# Set PROJECT_NAME based on benchmark
def set_project_name(benchmark):
    if benchmark == "widowx":
        return f"{WANDB_USERNAME}/tdqc-widowx"
    elif benchmark == "droid" or benchmark == "libero_pi0":
        return f"{WANDB_USERNAME}/tdqc-pi0"
    else:
        return f"{WANDB_USERNAME}/tdqc"

# Methods to compare - LIBERO
BASELINE_METHODS_LIBERO = [
    # "openvla-10-indep-mlp_clean",
    "openvla-10-indep-mlp_MSELoss",
    "openvla-10-lstm-lstm_clean",
    "openvla-10-lstm-lstm_TD0",
]

Q_LEARNING_METHODS_LIBERO = [
    "openvla-10-q_learning-top_k_probs",
]

# Methods to compare - WidowX
BASELINE_METHODS_WIDOWX = [
    # "openvla-widowx-indep-mlp_best",
    "openvla-widowx-indep-mlp_MSELoss_best",
    "openvla-widowx-lstm-lstm_best",
    "openvla-widowx-lstm-lstm_TD0_best"
]
Q_LEARNING_METHODS_WIDOWX = [
    "openvla-widowx-q_learning-widowx_top_k_probs_TD0_best",
]
BASELINE_METHODS_DRIOD = [
    "pizero_fast_droid-0510-indep-mlp_BCELoss_best",
    "pizero_fast_droid-0510-lstm-lstm_best2",

]
Q_LEARNING_METHODS_DRIOD = [
    "pizero_fast_droid-0510-lstm-13task_logits_best2"
]

BASELINE_METHODS_PI0_LIBERO = [
    "pizero_fast-default-lstm-lstm_pizero_FAST_Qiao_data",
    "pizero_fast-default-indep-mlp_pizero_FAST_Qiao_data",

]
Q_LEARNING_METHODS_PI0_LIBERO = [
    "pizero_fast-default-lstm-lstm_pizero_FAST_Qiao_data_logits_best"
]

def get_methods_for_benchmark(benchmark):
    if benchmark.lower() == "widowx":
        baseline = BASELINE_METHODS_WIDOWX
        qlearning = Q_LEARNING_METHODS_WIDOWX
        all_methods = baseline + qlearning
        benchmark_title = "openVLA-WidowX"
    elif benchmark.lower() == "droid":
        baseline = BASELINE_METHODS_DRIOD
        qlearning = Q_LEARNING_METHODS_DRIOD
        all_methods = baseline + qlearning
        benchmark_title = "pizero-fast-droid"
    elif benchmark.lower() == "libero_pi0":
        baseline = BASELINE_METHODS_PI0_LIBERO
        qlearning = Q_LEARNING_METHODS_PI0_LIBERO
        all_methods = baseline + qlearning
        benchmark_title = "Pi0-FAST-Libero"
    else:
        baseline = BASELINE_METHODS_LIBERO
        qlearning = Q_LEARNING_METHODS_LIBERO
        all_methods = baseline + qlearning
        benchmark_title = "openVLA-LIBERO"
    return baseline, qlearning, all_methods, benchmark_title

# Quantile metrics to extract
QUANTILES = [0.25, 0.5, 0.75, 1.0]
SPLITS = ['val_seen', 'val_unseen']  # 'train' commented out
END_POINT_X = 1.1  # X-axis position for end point metric


def filter_runs(run_name: str, project_name: str):
    """Filter runs matching the run name with seed suffixes."""
    all_runs = api.runs(project_name)
    runs = []
    for r in all_runs:
        if r.name.startswith(run_name):
            suffix = r.name[len(run_name):]
            if not suffix or (suffix.startswith('-') and suffix[1:].split('-')[0].isdigit()):
                try:
                    seed = r.config.get('train', {}).get('seed', 0)
                    if seed <= 20:
                        runs.append(r)
                except:
                    # If config is not accessible, include the run
                    runs.append(r)
    return runs

def extract_metrics(run_name: str, project_name: str, metric_type: str):
    """Extract quantile metrics for a method.
    
    Args:
        run_name: Name of the run to extract metrics for
        project_name: W&B project name
        metric_type: Either 'qt_all' or 'qt_all_max'
    """
    runs = filter_runs(run_name, project_name)
    print(f"  {run_name}: {len(runs)} runs")

    results = []
    for run in runs:
        summary = run.summary._json_dict
        row = {'method': run_name}

        for split in SPLITS:
            # Get quantile metrics
            for q in QUANTILES:
                # metric = f"falert_end_roc_auc/model_{split}_qt_{q}_{metric_type}"
                metric = f"roc_auc/model_{split}_tq{q}"
                if metric in summary:
                    row[f"{split}_q{q}"] = summary[metric]

            # Get end point metric from falert_end_roc_auc_taskwise
            # end_metric = f"falert_end_roc_auc_taskwise/model_{split}"
            # if end_metric in summary:
            #     if summary[end_metric] == 'Nan':
            #         row[f"{split}_end"] = 0
            #     else:
            #         row[f"{split}_end"] = summary[end_metric]

        results.append(row)

    return pd.DataFrame(results)


def plot_quantiles(df: pd.DataFrame, baseline_methods, qlearning_methods, all_methods, benchmark_title, metric_type: str, output='quantile_roc_auc.png'):
    """Plot ROC-AUC by quantiles for all methods.
    
    Args:
        df: DataFrame with metrics
        baseline_methods: List of baseline method names
        qlearning_methods: List of Q-learning method names
        all_methods: List of all method names
        benchmark_title: Title for the benchmark
        metric_type: Either 'qt_all' or 'qt_all_max'
        output: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("whitegrid")

    # Colors
    baseline_colors = ['#2E86AB', '#94C05B','#06A77D']
    qlearning_colors = [ '#820263','#D90368', '#F18F01','#4361EE']
    
    color_map = {}
    for i, method in enumerate(baseline_methods):
        color_map[method] = baseline_colors[i % len(baseline_colors)]
    for i, method in enumerate(qlearning_methods):
        color_map[method] = qlearning_colors[i % len(qlearning_colors)]

    for idx, split in enumerate(SPLITS):
        ax = axes[idx]

        for method in all_methods:
            method_df = df[df['method'] == method]

            # Extract quantile values
            quantile_cols = [f"{split}_q{q}" for q in QUANTILES]
            end_col = f"{split}_end"

            if not all(col in method_df.columns for col in quantile_cols):
                continue

            means = [method_df[col].mean() for col in quantile_cols]
            stds = [method_df[col].std() for col in quantile_cols]
            x_vals = list(QUANTILES)

            # Add end point metric if available
            # if end_col in method_df.columns and not method_df[end_col].isna().all():
            #     end_mean = method_df[end_col].mean()
            #     end_std = method_df[end_col].std()
            #     means.append(end_mean)
            #     stds.append(end_std)
            #     x_vals.append(END_POINT_X)

            # Plot
            label = method.replace('openvla-10-', '').replace('openvla-widowx-', '').replace('_', ' ').replace('MSELoss', 'BCE Loss')
            color = color_map.get(method)
            linestyle = '-' if method in baseline_methods else '--'

            ax.plot(x_vals, means, label=label, color=color,
                   linestyle=linestyle, linewidth=2, marker='o', markersize=4)
            ax.fill_between(x_vals,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2, color=color)

        ax.set_xlabel('Time Quantile from min task time', fontsize=11)
        ax.set_ylabel('ROC-AUC', fontsize=11)
        ax.set_title(split.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, 1.05)

        # Set x-ticks to include end point
        # xticks = list(QUANTILES) + [END_POINT_X]
        # xticklabels = [f'{q:.2f}' for q in QUANTILES] + ['end']
        xticks = list(QUANTILES)
        xticklabels = [f'{q:.2f}' for q in QUANTILES]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)

    # Add metric type to title
    metric_label = metric_type.replace('_', ' ').upper()
    fig.suptitle(f'{benchmark_title}: ROC-AUC by Time Quantile', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output}")
    plt.show()


def main(args):
    project_name = set_project_name(args.benchmark)
    baseline_methods, qlearning_methods, all_methods, benchmark_title = get_methods_for_benchmark(args.benchmark)
    
    # Process both metric types
    for metric_type in ['all', 'all_max']:
        print(f"\n{'='*80}")
        print(f"Processing metric type: qt_{metric_type}")
        print(f"{'='*80}")
        print(f"Extracting quantile metrics from {project_name}...\n")

        all_dfs = []
        for method in all_methods:
            df = extract_metrics(method, project_name, metric_type)
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Print summary
        print("\nSummary Statistics:")
        for split in SPLITS:
            print(f"\n{split.upper()}:")
            for method in all_methods:
                method_df = combined_df[combined_df['method'] == method]
                label = method.replace('openvla-10-', '').replace('openvla-widowx-', '').replace('_', ' ').replace('MSELoss', 'BCE Loss')
                print(f"  {label}")
                for q in QUANTILES:
                    col = f"{split}_q{q}"
                    if col in method_df.columns:
                        mean = method_df[col].mean()
                        std = method_df[col].std()
                        print(f"    q={q}: {mean:.4f} ± {std:.4f}")
                # Print end point metric
                # end_col = f"{split}_end"
                # if end_col in method_df.columns and not method_df[end_col].isna().all():
                #     mean = method_df[end_col].mean()
                #     std = method_df[end_col].std()
                #     print(f"    end: {mean:.4f} ± {std:.4f}")

        # Save to CSV
        output_csv = f'quantile_roc_auc_{args.benchmark}_{metric_type}.csv'
        combined_df.to_csv(output_csv, index=False)
        print(f"\nSaved data to: {output_csv}")

        # Plot
        output_plot = f'quantile_roc_auc_{args.benchmark}_{metric_type}.png'
        plot_quantiles(combined_df, baseline_methods, qlearning_methods, all_methods, benchmark_title, metric_type, output=output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract quantile ROC-AUC metrics from W&B and plot them"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="droid",
        choices=["libero", "widowx", "droid", "libero_pi0"],
        help="Benchmark to analyze: libero, widowx, droid, or libero_pi0",
    )
    args = parser.parse_args()
    main(args)

