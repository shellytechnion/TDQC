"""
Extract and plot metrics for TDQC variant comparison.
Compares different TDQC configurations (TD-0, TD-Lambda, top-k vs all probs).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

# Initialize the API
api = wandb.Api(timeout=60)
PROJECT_NAME = "anonymous/tdqc"

# Run names to extract
RUN_NAMES = [
    "openvla-10-q_learning-top_k_probs_TD0",
    "openvla-10-q_learning-top_k_probs_only_TD0_categorical",
    "openvla-10-q_learning-probs_only_TDLambda_05",
    "openvla-10-q_learning-probs_only_TD0",
    "openvla-10-q_learning-top_k_probs_lambda",
    "openvla-10-q_learning-probs_only_TDLambda",
]

# Label mappings for plot legends
LABEL_MAPPING = {
    "openvla-10-q_learning-top_k_probs_TD0": "Top-10 Probs TD-0",
    "openvla-10-q_learning-top_k_probs_only_TD0_categorical": "Top-10 Probs TD-0 (categorical)",
    "openvla-10-q_learning-probs_only_TDLambda_05": "Max Probs TD-λ (λ=0.5)",
    "openvla-10-q_learning-probs_only_TD0": "Max Probs TD-0",
    "openvla-10-q_learning-top_k_probs_lambda": "Top-10 Probs TD-λ",
    "openvla-10-q_learning-probs_only_TDLambda": "Max Probs TD-λ (λ=0.8)",
}

# Color scheme
COLORS = {
    "openvla-10-q_learning-top_k_probs_TD0": "#820263",
    "openvla-10-q_learning-top_k_probs_only_TD0_categorical": "#E06CBE",
    "openvla-10-q_learning-probs_only_TDLambda_05": "#D90368",
    "openvla-10-q_learning-probs_only_TD0": "#F18F01",
    "openvla-10-q_learning-top_k_probs_lambda": "#4361EE",
    "openvla-10-q_learning-probs_only_TDLambda": "#2EC4B6",
}

# Metrics to extract
METRICS = [
    "falert_early_roc_auc/model_val_seen",
    "falert_early_roc_auc/model_val_unseen",
    "calibration/model_ece_at_stop_val_seen",
    "calibration/model_ece_at_stop_val_unseen",
    "calibration/model_brier_at_stop_val_seen",
    "calibration/model_brier_at_stop_val_unseen",
]


def get_runs(project_name: str):
    """Fetch runs from W&B API."""
    print("Fetching runs from W&B API...")
    runs = list(api.runs(project_name))
    print(f"Fetched {len(runs)} runs")
    return runs


def filter_runs_by_name(all_runs, run_name: str):
    """Filter runs by name with seed suffix, excluding runs with seed > 20."""
    filtered_runs = []
    for r in all_runs:
        if r.name.startswith(run_name) and (
            (suffix := r.name[len(run_name):]) == '' or
            (suffix.startswith('-') and suffix[1:].split('-')[0].isdigit())
        ):
            # Check seed from config
            try:
                seed = r.config.get('train', {}).get('seed', 0)
                if seed <= 20:
                    filtered_runs.append(r)
                    print(f"  Including run: {r.name} (seed={seed})")
                else:
                    print(f"  Excluding run: {r.name} (seed={seed} > 20)")
            except:
                # If config is not accessible, include the run
                filtered_runs.append(r)
                print(f"  Including run: {r.name} (config not accessible)")
    return filtered_runs


def extract_metrics_for_run_name(project_name: str, run_name: str, metrics: list) -> dict:
    """Extract metrics for all runs matching the given run name."""
    all_runs = get_runs(project_name)
    runs = filter_runs_by_name(all_runs, run_name)
    
    print(f"\nProcessing: {run_name}")
    print(f"  Found {len(runs)} matching runs with seed <= 20")
    
    # Pre-allocate results
    results = {metric: [] for metric in metrics}
    
    # Batch process summaries
    for run in runs:
        summary = run.summary._json_dict
        for metric in metrics:
            if metric in summary:
                results[metric].append(summary[metric])
    
    # Print extracted values
    for metric in metrics:
        values = results[metric]
        if values:
            metric_short = metric.split('/')[-1]
            print(f"  {metric_short}: {np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})")
    
    return results


def plot_scatter_comparison(all_data: dict, metric_key: str, metric_label: str, output_path: str):
    """
    Create scatter plot comparing all methods for a specific metric with error bars.
    
    Args:
        all_data: Dictionary with run_name -> metric_name -> list of values
        metric_key: Substring to identify the metric (e.g., 'brier', 'ece', 'roc_auc')
        metric_label: Label for y-axis
        output_path: Path to save the plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # Create figure with 2 subplots (seen and unseen)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, split in [(ax1, 'val_seen'), (ax2, 'val_unseen')]:
        plot_data = []
        
        for run_name in RUN_NAMES:
            if run_name not in all_data:
                continue
            
            # Find metric with matching key and split
            matching_metrics = [m for m in all_data[run_name].keys() 
                              if metric_key in m and split in m]
            
            if not matching_metrics:
                continue
            
            metric_name = matching_metrics[0]
            values = all_data[run_name][metric_name]
            
            if not values:
                continue
            
            plot_data.append({
                'method': LABEL_MAPPING.get(run_name, run_name),
                'method_full': run_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            })
        
        if not plot_data:
            continue
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create scatter plot with error bars
        x_positions = np.arange(len(df_plot))
        colors = [COLORS.get(row['method_full'], '#999999') for _, row in df_plot.iterrows()]
        
        # Plot points with error bars
        for i, row in df_plot.iterrows():
            ax.errorbar(i, row['mean'], yerr=row['std'],
                       fmt='o', markersize=12, capsize=8, capthick=2, 
                       color=colors[i], markeredgecolor='black', markeredgewidth=1.5,
                       elinewidth=2, alpha=0.85, label=row['method'])
        
        # Customize plot
        ax.set_ylabel(metric_label, fontsize=20)
        split_label = 'Seen' if split == 'val_seen' else 'Unseen'
        ax.set_title(f'{split_label} Tasks', fontsize=22, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df_plot['method'], rotation=45, ha='right', fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=6)
        
        # Set y-axis limits based on metric type with some padding
        if 'ECE' in metric_label or 'Brier' in metric_label:
            y_min = max(0, df_plot['mean'].min() - df_plot['std'].max() - 0.02)
            y_max = min(0.5, df_plot['mean'].max() + df_plot['std'].max() + 0.02)
            ax.set_ylim(bottom=y_min, top=y_max)
        elif 'ROC' in metric_label:
            y_min = max(0, df_plot['mean'].min() - df_plot['std'].max() - 0.05)
            y_max = min(1.0, df_plot['mean'].max() + df_plot['std'].max() + 0.05)
            ax.set_ylim(bottom=y_min, top=y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot metrics for TDQC variant comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots_tdqc_variants",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="./scripts/tdqc_variants_results.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract data for all run names
    all_data = {}
    all_results = []
    
    for run_name in RUN_NAMES:
        metrics_data = extract_metrics_for_run_name(PROJECT_NAME, run_name, METRICS)
        all_data[run_name] = metrics_data
        
        # Build row for CSV
        row = {"run_name": run_name, "label": LABEL_MAPPING.get(run_name, run_name)}
        
        for metric in METRICS:
            values = metrics_data[metric]
            metric_short = metric.split('/')[-1]
            
            row[f"{metric_short}_mean"] = np.mean(values) if values else np.nan
            row[f"{metric_short}_std"] = np.std(values) if values else np.nan
            row[f"{metric_short}_n"] = len(values)
        
        all_results.append(row)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    cols = ["run_name", "label"]
    for metric in METRICS:
        metric_short = metric.split('/')[-1]
        cols.extend([f"{metric_short}_mean", f"{metric_short}_std", f"{metric_short}_n"])
    
    df = df[cols]
    df.to_csv(args.csv_output, index=False)
    print(f"\nSaved results to: {args.csv_output}")
    
    # Create plots
    print("\n" + "=" * 100)
    print("CREATING PLOTS")
    print("=" * 100)
    
    # Plot Brier Score
    plot_scatter_comparison(
        all_data,
        metric_key='brier',
        metric_label='Brier Score at Stop Time',
        output_path=os.path.join(args.output_dir, 'tdqc_variants_brier.png')
    )
    
    # Plot ECE
    plot_scatter_comparison(
        all_data,
        metric_key='ece',
        metric_label='ECE at Stop Time',
        output_path=os.path.join(args.output_dir, 'tdqc_variants_ece.png')
    )
    
    # Plot ROC-AUC
    plot_scatter_comparison(
        all_data,
        metric_key='roc_auc',
        metric_label='ROC-AUC',
        output_path=os.path.join(args.output_dir, 'tdqc_variants_roc_auc.png')
    )
    
    print("\nAll plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
