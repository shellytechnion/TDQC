"""
Average calibration curves across multiple benchmarks.

This script reads calibration curve data from multiple experiment folders
(plots_libero, plots_droid, plots_pi0_libero, plots_widowx) and computes
the average calibration metrics across all benchmarks.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


# Experiment folders to process
EXPERIMENT_FOLDERS = [
    "plots_libero",
    "plots_droid", 
    "plots_pi0_libero",
    "plots_widowx"
]

# CSV files to process
CALIBRATION_FILES = [
    "calibration_curves_model_brier_val_unseen_table_data.csv",
    "calibration_curves_model_brier_val_seen_table_data.csv",
    "calibration_curves_model_ece_val_unseen_table_data.csv",
    "calibration_curves_model_ece_val_seen_table_data.csv",
]

# Method label mapping for consistency
METHOD_LABEL_MAPPING = {
    "MLP (BCE Loss)": "MLP (BCE Loss)",
    "MLP (TD-0)": "MLP (TD-0)",
    "LSTM (BCE Loss)": "LSTM (BCE Loss)",
    "LSTM (TD-0)": "LSTM (TD-0)",
    "TDQC (Top-10 BCE Loss)": "TDQC (Top-10 BCE Loss)",
    "TDQC (Top-10 TD-0)": "TDQC (Top-10 TD-0)",
}

# Color grouping: methods with the same base get the same color
COLOR_GROUPS = {
    "mlp": {
        "color": "#2E86AB",
        "methods": ["MLP (BCE Loss)", "MLP (TD-0)"]
    },
    "lstm": {
        "color": "#94C05B",
        "methods": ["LSTM (BCE Loss)", "LSTM (TD-0)"]
    },
    "qlearning": {
        "color": "#820263",
        "methods": ["TDQC (Top-10 BCE Loss)", "TDQC (Top-10 TD-0)"]
    },
}


def get_method_color_and_style(method_label: str) -> tuple:
    """
    Get color and linestyle for a method.
    
    Returns:
        tuple: (color, linestyle) where linestyle is '-' for first method in group, '--' for others
    """
    for group_name, group_info in COLOR_GROUPS.items():
        if method_label in group_info["methods"]:
            color = group_info["color"]
            method_idx = group_info["methods"].index(method_label)
            linestyle = '-' if method_idx == 0 else '--'
            return color, linestyle
    
    # Fallback
    return '#999999', '-'


def load_calibration_data(base_dir: str, experiment_folders: list, filename: str) -> pd.DataFrame:
    """
    Load calibration data from multiple experiment folders.
    
    Args:
        base_dir: Base directory containing experiment folders
        experiment_folders: List of experiment folder names
        filename: Name of the CSV file to load
    
    Returns:
        Combined DataFrame with all data, including a 'benchmark' column
    """
    all_data = []
    
    for folder in experiment_folders:
        filepath = os.path.join(base_dir, folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        try:
            df = pd.read_csv(filepath)
            df['benchmark'] = folder.replace('plots_', '')
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {folder}/{filename}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not all_data:
        raise ValueError(f"No data loaded for {filename}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows loaded: {len(combined_df)}")
    
    return combined_df


def average_calibration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average calibration data across benchmarks.
    
    Groups by method_label, time_quantile, and split, then averages the metric values
    across all benchmarks.
    
    Args:
        df: Combined DataFrame with data from all benchmarks
    
    Returns:
        DataFrame with averaged metrics
    """
    # Filter out TDQC methods
    df = df[~df['method_label'].str.contains('TDQI', case=False, na=False)]
    
    # Determine metric column name
    if 'brier_mean' in df.columns:
        metric_col = 'brier'
    elif 'ece_mean' in df.columns:
        metric_col = 'ece'
    else:
        raise ValueError(f"Unknown metric columns: {df.columns.tolist()}")
    
    mean_col = f'{metric_col}_mean'
    std_col = f'{metric_col}_std'
    
    # Group by method_label, time_quantile, and split
    grouped = df.groupby(['method_label', 'time_quantile', 'split']).agg({
        mean_col: 'mean',  # Average the means across benchmarks
        std_col: lambda x: np.sqrt(np.mean(x**2)),  # RMS of stds (conservative estimate)
        'n_samples': 'sum',  # Total number of samples
        'benchmark': 'count'  # Number of benchmarks contributing
    }).reset_index()
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        mean_col: f'{metric_col}_mean_avg',
        std_col: f'{metric_col}_std_avg',
        'benchmark': 'n_benchmarks'
    })
    
    # Add metric type column
    grouped['metric_type'] = metric_col
    
    return grouped


def plot_averaged_calibration_curves(df: pd.DataFrame, output_dir: str, metric_type: str):
    """
    Plot averaged calibration curves.
    
    Args:
        df: DataFrame with averaged calibration data
        output_dir: Directory to save plots
        metric_type: Type of metric ('brier' or 'ece')
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Plot for each split
    for split in df['split'].unique():
        split_df = df[df['split'] == split].copy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique methods
        methods = split_df['method_label'].unique()
        
        for method_label in methods:
            method_df = split_df[split_df['method_label'] == method_label]
            
            if len(method_df) == 0:
                continue
            
            # Sort by time_quantile
            method_df = method_df.sort_values('time_quantile')
            
            color, linestyle = get_method_color_and_style(method_label)
            mean_col = f'{metric_type}_mean_avg'
            std_col = f'{metric_type}_std_avg'
            
            # Plot line with error bars
            ax.plot(
                method_df['time_quantile'],
                method_df[mean_col],
                label=method_label,
                linewidth=2.5,
                color=color,
                linestyle=linestyle
            )
            
            # Add shaded region for std
            ax.fill_between(
                method_df['time_quantile'],
                method_df[mean_col] - method_df[std_col],
                method_df[mean_col] + method_df[std_col],
                alpha=0.2,
                color=color
            )
        
        # Customize plot
        metric_name = 'ECE' if metric_type == 'ece' else 'Brier Score'
        ax.set_xlabel('Time Quantile', fontsize=16)
        ax.set_ylabel(f'{metric_name} (Averaged)', fontsize=16)
        ax.set_title(
            f'Averaged {metric_name} Across All Benchmarks - {split.replace("_", " ").title()}',
            fontsize=18,
            fontweight='bold'
        )
        ax.legend(loc='best', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=0.5)
        ax.set_xlim(left=-0.02, right=1)
        
        # Save plot
        plot_filename = f"averaged_calibration_{metric_type}_{split}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()


def plot_bar_at_selected_times(df: pd.DataFrame, output_dir: str, metric_type: str, 
                                selected_times: list = None):
    """
    Plot bar charts at selected time quantiles.
    
    Args:
        df: DataFrame with averaged calibration data
        output_dir: Directory to save plots
        metric_type: Type of metric ('brier' or 'ece')
        selected_times: List of time quantiles to plot (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    """
    if selected_times is None:
        selected_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    mean_col = f'{metric_type}_mean_avg'
    std_col = f'{metric_type}_std_avg'
    
    # Plot for each split
    for split in df['split'].unique():
        split_df = df[df['split'] == split].copy()
        
        # Filter to selected times (with tolerance)
        plot_data = []
        for time_point in selected_times:
            time_df = split_df[np.abs(split_df['time_quantile'] - time_point) < 0.05]
            if len(time_df) > 0:
                # Take closest value for each method
                for method in time_df['method_label'].unique():
                    method_df = time_df[time_df['method_label'] == method]
                    closest_idx = (method_df['time_quantile'] - time_point).abs().idxmin()
                    plot_data.append(method_df.loc[closest_idx].copy())
        
        if not plot_data:
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get unique methods and times
        methods = plot_df['method_label'].unique()
        n_methods = len(methods)
        n_times = len(selected_times)
        bar_width = 0.8 / n_methods
        
        for i, method_label in enumerate(methods):
            method_data = plot_df[plot_df['method_label'] == method_label]
            
            if len(method_data) == 0:
                continue
            
            # Calculate x positions for available times
            x_pos = []
            means = []
            stds = []
            
            for j, time_point in enumerate(selected_times):
                time_data = method_data[np.abs(method_data['time_quantile'] - time_point) < 0.05]
                if len(time_data) > 0:
                    x_pos.append(j + i * bar_width)
                    means.append(time_data[mean_col].values[0])
                    stds.append(time_data[std_col].values[0])
            
            if means:
                color, linestyle = get_method_color_and_style(method_label)
                hatch = '//' if linestyle == '--' else ''
                ax.bar(
                    x_pos, means, bar_width,
                    yerr=stds,
                    label=method_label,
                    color=color,
                    alpha=0.8,
                    capsize=3,
                    hatch=hatch
                )
        
        # Customize plot
        metric_name = 'ECE' if metric_type == 'ece' else 'Brier Score'
        ax.set_xlabel('Time Quantile', fontsize=16)
        ax.set_ylabel(f'{metric_name} (Averaged)', fontsize=16)
        ax.set_title(
            f'Averaged {metric_name} at Selected Times - {split.replace("_", " ").title()}',
            fontsize=18,
            fontweight='bold'
        )
        ax.set_xticks(np.arange(n_times) + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels([f'{t:.1f}' for t in selected_times], fontsize=14)
        ax.tick_params(axis='y', which='major', labelsize=14)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0, top=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"averaged_calibration_{metric_type}_{split}_bars.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar plot: {plot_path}")
        plt.close()


def main(args: argparse.Namespace):
    """Main function to average calibration data and create plots."""
    
    print("=" * 100)
    print("AVERAGING CALIBRATION CURVES ACROSS BENCHMARKS")
    print("=" * 100)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each calibration file
    for filename in CALIBRATION_FILES:
        print(f"\nProcessing: {filename}")
        print("-" * 100)
        
        # Load data from all benchmarks
        try:
            combined_df = load_calibration_data(args.base_dir, EXPERIMENT_FOLDERS, filename)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")
            continue
        
        # Average across benchmarks
        averaged_df = average_calibration_data(combined_df)
        print(f"Averaged data shape: {averaged_df.shape}")
        print(f"Methods: {averaged_df['method_label'].unique().tolist()}")
        print(f"Splits: {averaged_df['split'].unique().tolist()}")
        
        # Determine metric type
        if 'brier' in filename:
            metric_type = 'brier'
        elif 'ece' in filename:
            metric_type = 'ece'
        else:
            print(f"Unknown metric type for {filename}, skipping plots")
            continue
        
        # Save averaged data to CSV
        output_csv = os.path.join(args.output_dir, f"averaged_{filename}")
        averaged_df.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"Saved averaged data: {output_csv}")
        
        # Create plots
        if args.plot:
            print("\nCreating plots...")
            plot_averaged_calibration_curves(averaged_df, args.output_dir, metric_type)
            
            if args.plot_bars:
                plot_bar_at_selected_times(averaged_df, args.output_dir, metric_type)
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average calibration curves across multiple benchmarks"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing experiment folders (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots_averaged",
        help="Directory to save averaged results and plots",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate calibration curve plots",
    )
    parser.add_argument(
        "--plot-bars",
        action="store_true",
        default=True,
        help="Generate bar plots at selected times",
    )
    
    args = parser.parse_args()
    main(args)
