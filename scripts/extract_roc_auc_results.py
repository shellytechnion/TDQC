"""
Extract ROC AUC results and calibration curves for specific run names from wandb.
example:
python ./scripts/extract_roc_auc_results.py --output ./plots_droid --plot-dir ./plots_droid --benchmark droid --output roc_auc_results_pi0_fast_droid.csv

"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
import time
from scipy import stats
from itertools import combinations
from collections import defaultdict

# Initialize the API with longer timeout
api = wandb.Api(timeout=80)
WANDB_USERNAME = api.viewer.username

# Set PROJECT_NAME based on benchmark
def set_project_name(benchmark):
    return f"{WANDB_USERNAME}/tdqc-final"

def get_runs(project_name: str):
    """Fetch runs from W&B API."""
    print("Fetching runs from W&B API (this may take a moment)...")
    runs = list(api.runs(project_name))
    print(f"Fetched {len(runs)} runs")
    return runs

# Run names to extract (each has 5 runs inside)
# Baseline methods
BASELINE_METHODS_LIBERO = [
    # "openvla-10-lstm-lstm_images_res18",
    # "openvla-10-lstm-lstm_TD0_top_k_probs_2000",
    # "openvla-10-lstm-lstm_TD0_top_k_probs_GRU",
    "openvla-10-indep-mlp_BCE",
    "openvla-10-indep-mlp_TD0",
    "openvla-10-lstm-lstm",
    "openvla-10-lstm-lstm_TD0",
]
Q_LEARNING_METHODS_LIBERO = [
    "openvla-10-q_learning-top_k_probs_BCE",
    "openvla-10-q_learning-top_k_probs_TD0",
]
BASELINE_METHODS_WIDOWX = [
    "openvla-widowx-indep-mlp_BCE",
    "openvla-widowx-indep-mlp_TD0",
    "openvla-widowx-lstm-lstm",
    "openvla-widowx-lstm-lstm_TD0"
]
Q_LEARNING_METHODS_WIDOWX = [
    "openvla-widowx-q_learning-top_k_probs_BCE",
    "openvla-widowx-q_learning-top_k_probs",
]

BASELINE_METHODS_DRIOD = [
    "pizero_fast_droid-0510-indep-mlp_BCE",
    "pizero_fast_droid-0510-indep-mlp_TD0",
    "pizero_fast_droid-0510-lstm-lstm",
    "pizero_fast_droid-0510-lstm-lstm_TD0"

]
Q_LEARNING_METHODS_DRIOD = [
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0"
]

BASELINE_METHODS_PI0_FAST_LIBERO = [
    "pizero_fast-default-indep-mlp_BCE",
    "pizero_fast-default-indep-mlp_TD0",
    "pizero_fast-default-lstm-lstm",
    "pizero_fast-default-lstm-lstm_TD0"
]
Q_LEARNING_METHODS_PI0_FAST_LIBERO = [
    "pizero_fast-default-lstm-lstm_top_k_probs_BCE",
    "pizero_fast-default-lstm-lstm_top_k_probs_TD0",
]

BASELINE_METHODS_PI0_LIBERO = [
    "pizero-default-indep-mlp_BCE",
    "pizero-default-indep-mlp_TD0",
    "pizero-default-lstm-lstm",
    "pizero-default-lstm-lstm_TD0"
]

BASELINE_METHODS_UNIVLA = [
    # "univla-default-indep-mlp",
    "univla-default-indep-mlp_BCE",
    "univla-default-indep-mlp_TD0",
    "univla-default-lstm-lstm_BCE",
    "univla-default-lstm-lstm_TD0"
]
Q_LEARNING_METHODS_UNIVLA = [
    "univla-default-lstm-lstm_top_k_probs_BCE",
    "univla-default-lstm-lstm_top_k_probs_TD0"
]


# Label mappings for plot legends
LABEL_MAPPING = {
    # LIBERO methods
    "openvla-10-indep-mlp_BCE": "MLP (BCE Loss)",
    "openvla-10-indep-mlp_TD0": "MLP (TD-0)",
    "openvla-10-lstm-lstm": "LSTM (BCE Loss)",
    "openvla-10-lstm-lstm_TD0": "LSTM (TD-0)", 
    "openvla-10-q_learning-top_k_probs_BCE": "TDQC (Top-10 BCE Loss)",
    "openvla-10-q_learning-top_k_probs_TD0": "TDQC (Top-10 TD-0)",
    # "openvla-10-q_learning-top_k_probs_TD0_best_2seeds": "TDQC (Top-10 TD-0)",
    
    # WidowX methods
    "openvla-widowx-indep-mlp_BCE": "MLP (BCE Loss)",
    "openvla-widowx-indep-mlp_TD0": "MLP (TD-0)",
    "openvla-widowx-lstm-lstm": "LSTM (BCE Loss)",
    "openvla-widowx-lstm-lstm_TD0": "LSTM (TD-0)",
    "openvla-widowx-q_learning-top_k_probs_BCE": "TDQC (Top-10 BCE Loss)",
    "openvla-widowx-q_learning-top_k_probs": "TDQC (Top-10 TD-0)",
    
    # Droid methods
    "pizero_fast_droid-0510-indep-mlp_BCE": "MLP (BCE Loss)",
    "pizero_fast_droid-0510-indep-mlp_TD0": "MLP (TD-0)",
    "pizero_fast_droid-0510-lstm-lstm": "LSTM (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_TD0": "LSTM (TD-0)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE": "TDQC (Top-10 BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0": "TDQC (Top-10 TD-0)",


    # Pi0 FAST LIBERO methods
    "pizero_fast-default-indep-mlp_BCE" : "MLP (BCE Loss)",
    "pizero_fast-default-indep-mlp_TD0": "MLP (TD-0)",
    "pizero_fast-default-lstm-lstm": "LSTM (BCE Loss)",
    "pizero_fast-default-lstm-lstm_TD0": "LSTM (TD-0)",
    "pizero_fast-default-lstm-lstm_top_k_probs_BCE": "TDQC (Top-10 BCE Loss)",
    "pizero_fast-default-lstm-lstm_top_k_probs_TD0": "TDQC (Top-10 TD-0)",

    #Pi0 LIBERO methods
    "pizero-default-indep-mlp_BCE": "MLP (BCE Loss)",
    "pizero-default-indep-mlp_TD0": "MLP (TD-0)",
    "pizero-default-lstm-lstm":"LSTM (BCE Loss)",
    "pizero-default-lstm-lstm_TD0": "LSTM (TD-0)",

    # UniVLA methods
    "univla-default-indep-mlp_BCE": "MLP (BCE Loss)",
    "univla-default-indep-mlp_TD0": "MLP (TD-0)",
    "univla-default-lstm-lstm_BCE": "LSTM (BCE Loss)",
    "univla-default-lstm-lstm_TD0": "LSTM (TD-0)",
    "univla-default-lstm-lstm_top_k_probs_BCE": "TDQC (Top-10 BCE Loss)",
    "univla-default-lstm-lstm_top_k_probs_TD0": "TDQC (Top-10 TD-0)",
}

# Color grouping: methods with the same base get the same color
# Format: {base_name: (color, [list_of_methods])}
COLOR_GROUPS = {
    # LIBERO
    "libero_mlp": {
        "color": "#2E86AB",
        "methods": ["openvla-10-indep-mlp_BCE", "openvla-10-indep-mlp_TD0"]
    },
    "libero_lstm": {
        "color": "#94C05B",
        "methods": ["openvla-10-lstm-lstm", "openvla-10-lstm-lstm_TD0"]
    },
    "libero_qlearning": {
        "color": "#820263",
        "methods": ["openvla-10-q_learning-top_k_probs_BCE", "openvla-10-q_learning-top_k_probs_TD0"]
    },
    
    # WidowX
    "widowx_mlp": {
        "color": "#2E86AB",
        "methods": ["openvla-widowx-indep-mlp_BCE", "openvla-widowx-indep-mlp_TD0"]
    },
    "widowx_lstm": {
        "color": "#94C05B",
        "methods": ["openvla-widowx-lstm-lstm", "openvla-widowx-lstm-lstm_TD0"]
    },
    "widowx_qlearning": {
        "color": "#820263",
        "methods": ["openvla-widowx-q_learning-top_k_probs_BCE", "openvla-widowx-q_learning-top_k_probs"]
    },
    
    # Droid
    "droid_mlp": {
        "color": "#2E86AB",
        "methods": ["pizero_fast_droid-0510-indep-mlp_BCE", "pizero_fast_droid-0510-indep-mlp_TD0"]
    },
    "droid_lstm": {
        "color": "#94C05B",
        "methods": ["pizero_fast_droid-0510-lstm-lstm", "pizero_fast_droid-0510-lstm-lstm_TD0"]
    },
    "droid_qlearning": {
        "color": "#820263",
        "methods": ["pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE", "pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0"]
    },

    # Pi0 FAST LIBERO
    "pi0_fast_mlp": {
        "color": "#2E86AB",
        "methods": [    "pizero_fast-default-indep-mlp_BCE",
                        "pizero_fast-default-indep-mlp_TD0",]
    },
    "pi0_fast_lstm": {
        "color": "#94C05B",
        "methods": [    "pizero_fast-default-lstm-lstm",
                        "pizero_fast-default-lstm-lstm_TD0",]
    },
    "pi0_fast_qlearning": {
        "color": "#820263",
        "methods": [    "pizero_fast-default-lstm-lstm_top_k_probs_BCE",
                        "pizero_fast-default-lstm-lstm_top_k_probs_TD0",]
    },
    
    # Pi0  LIBERO
    "pi0_mlp": {
        "color": "#2E86AB",
        "methods": [    "pizero-default-indep-mlp_BCE",
                        "pizero-default-indep-mlp_TD0",]
    },
    "pi0_lstm": {
        "color": "#94C05B",
        "methods": [    "pizero-default-lstm-lstm",
                        "pizero-default-lstm-lstm_TD0",]
    },

    # UniVLA
    "univla_mlp": {
        "color": "#2E86AB",
        "methods": [    "univla-default-indep-mlp_BCE",
                        "univla-default-indep-mlp_TD0",]
    },  
    "univla_lstm": {
        "color": "#94C05B",
        "methods": [    "univla-default-lstm-lstm_BCE",
                        "univla-default-lstm-lstm_TD0",]
    },
    "univla_qlearning": {
        "color": "#820263",
        "methods": [    "univla-default-lstm-lstm_top_k_probs_BCE",
                        "univla-default-lstm-lstm_top_k_probs_TD0",]
    },
}

def get_method_label(method_name: str) -> str:
    """Get the display label for a method."""
    return LABEL_MAPPING.get(method_name, method_name.replace('openvla-10-', '').replace('openvla-widowx-', '').replace('_', ' '))

def get_method_color_and_style(method_name: str, baseline_methods: list, qlearning_methods: list) -> tuple:
    """
    Get color and linestyle for a method.
    
    Returns:
        tuple: (color, linestyle). TD-0 variants are always dashed.
    """
    method_name_lower = method_name.lower()
    # Keep TD-0 visually consistent even when legacy names do not include explicit "TD0".
    is_td0_variant = (
        "td0" in method_name_lower
        or ("q_learning-top_k_probs" in method_name_lower and "bce" not in method_name_lower)
    )

    # Find which color group this method belongs to
    for group_name, group_info in COLOR_GROUPS.items():
        if method_name in group_info["methods"]:
            color = group_info["color"]
            method_idx = group_info["methods"].index(method_name)
            linestyle = '--' if is_td0_variant else ('-' if method_idx == 0 else '--')
            return color, linestyle
    
    # Fallback: use old color scheme
    baseline_colors = ['#2E86AB', '#94C05B', '#06A77D']
    qlearning_colors = ['#820263', '#D90368', '#F18F01', '#4361EE']
    
    if method_name in baseline_methods:
        idx = baseline_methods.index(method_name)
        return baseline_colors[idx % len(baseline_colors)], '--' if is_td0_variant else '-'
    elif method_name in qlearning_methods:
        idx = qlearning_methods.index(method_name)
        return qlearning_colors[idx % len(qlearning_colors)], '--' if is_td0_variant else '-'
    
    return '#999999', '-'
def get_methods_for_benchmark(benchmark):
    if benchmark.lower() == "widowx":
        return BASELINE_METHODS_WIDOWX, Q_LEARNING_METHODS_WIDOWX, BASELINE_METHODS_WIDOWX + Q_LEARNING_METHODS_WIDOWX, "openVLA-WidowX"
    elif benchmark.lower() == "droid":
        return BASELINE_METHODS_DRIOD, Q_LEARNING_METHODS_DRIOD, BASELINE_METHODS_DRIOD + Q_LEARNING_METHODS_DRIOD, "Pi0-FAST-Droid"
    elif benchmark.lower() == "libero_fast_pi0":
        return BASELINE_METHODS_PI0_FAST_LIBERO, Q_LEARNING_METHODS_PI0_FAST_LIBERO, BASELINE_METHODS_PI0_FAST_LIBERO + Q_LEARNING_METHODS_PI0_FAST_LIBERO, "Pi0-FAST-Libero"
    elif benchmark.lower() == "libero_pi0":
        return BASELINE_METHODS_PI0_LIBERO, Q_LEARNING_METHODS_PI0_LIBERO, BASELINE_METHODS_PI0_LIBERO + Q_LEARNING_METHODS_PI0_LIBERO, "Pi0-Libero"
    elif benchmark.lower() == "univla":
        return BASELINE_METHODS_UNIVLA, Q_LEARNING_METHODS_UNIVLA, BASELINE_METHODS_UNIVLA + Q_LEARNING_METHODS_UNIVLA, "UniVLA"
    else:
        return BASELINE_METHODS_LIBERO, Q_LEARNING_METHODS_LIBERO, BASELINE_METHODS_LIBERO + Q_LEARNING_METHODS_LIBERO, "openVLA-LIBERO"

# Metrics to extract
METRICS = [
    "falert_early_roc_auc/model_val_seen",
    "falert_early_roc_auc/model_val_unseen",
]
METRICS_DROID = [
    "falert_early_roc_auc_taskwise/model_val_seen",
    "falert_early_roc_auc_taskwise/model_val_unseen",
]
# Calibration metrics at stop (task_min_step - 1)
CALIBRATION_AT_STOP_METRICS = [
    "calibration/model_ece_at_stop_val_seen",
    "calibration/model_ece_at_stop_val_unseen",
    "calibration/model_brier_at_stop_val_seen",
    "calibration/model_brier_at_stop_val_unseen",
]

# Calibration curve tables to extract
CALIBRATION_TABLES = [
    "calibration_curves/model_ece_val_unseen_table",
    "calibration_curves/model_ece_val_seen_table",
    "calibration_curves/model_brier_val_unseen_table",
    "calibration_curves/model_brier_val_seen_table",
]


def filter_runs_by_name(all_runs, run_name: str):
    """Filter runs by name with seed suffix, excluding runs with seed > 20."""
    filtered_runs = []
    for r in all_runs:
        if r.name.startswith(run_name) and (
            (suffix := r.name[len(run_name):]) == '' or
            (suffix.startswith('-') and suffix[1:].split('-')[0].isdigit())
        ):
            # Exclude runs with seed > 20
            try:
                seed = r.config.get('train', {}).get('seed', 0)
                if seed <= 20:
                    filtered_runs.append(r)
            except:
                # If config is not accessible, include the run
                filtered_runs.append(r)
    return filtered_runs

def extract_metrics_for_run_name(project_name: str, run_name: str, metrics: list[str]) -> dict:
    """
    Extract metrics for all runs matching the given run name.
    
    Args:
        project_name: W&B project name
        run_name: Name/group to filter runs
        metrics: List of metric names to extract
    
    Returns:
        Dictionary with metric names as keys and lists of values as values
    """
    all_runs = get_runs(project_name)
    runs = filter_runs_by_name(all_runs, run_name)
    
    # Pre-allocate results
    results = {metric: [] for metric in metrics}
    
    # Batch process summaries
    for run in runs:
        summary = run.summary._json_dict
        for metric in metrics:
            if metric in summary:
                results[metric].append(summary[metric])
    
    return results


def extract_run_data(project_name: str, run_names: list[str], metrics: list[str] = None, table_names: list[str] = None) -> dict:
    """
    Unified function to extract both metrics and calibration tables for all runs matching the given run names.
    Fetches all runs once and extracts data for all run names in a single pass to minimize W&B API calls.
    
    Args:
        project_name: W&B project name
        run_names: List of run name/groups to filter runs
        metrics: List of metric names to extract (optional)
        table_names: List of table names to extract (optional)
    
    Returns:
        Dictionary mapping run_name -> {'metrics': {...}, 'tables': {...}}
    """
    import json
    import tempfile
    import shutil
    
    # Fetch all runs once
    all_runs = get_runs(project_name)
    
    # Initialize results for all run names
    results = {}
    for run_name in run_names:
        results[run_name] = {
            'metrics': {metric: [] for metric in (metrics or [])},
            'tables': {table_name: [] for table_name in (table_names or [])}
        }
    
    # Create single temp directory for all downloads
    temp_dir = tempfile.mkdtemp() if table_names else None
    
    try:
        # Single pass through all runs
        for run in all_runs:
            # Check which run_name(s) this run belongs to
            matching_run_names = []
            for run_name in run_names:
                if run.name.startswith(run_name) and (
                    (suffix := run.name[len(run_name):]) == '' or
                    (suffix.startswith('-') and suffix[1:].split('-')[0].isdigit())
                ):
                    # Exclude runs with seed > 20
                    try:
                        seed = run.config.get('train', {}).get('seed', 0)
                        if seed <= 20:
                            matching_run_names.append(run_name)
                    except:
                        matching_run_names.append(run_name)
            
            if not matching_run_names:
                continue
            
            summary = run.summary._json_dict
            
            # Extract data for all matching run names
            for run_name in matching_run_names:
                # Extract metrics
                if metrics:
                    for metric in metrics:
                        if metric in summary:
                            results[run_name]['metrics'][metric].append(summary[metric])
                
                # Extract calibration tables
                if table_names:
                    for table_name in table_names:
                        try:
                            if table_name not in summary:
                                continue
                            
                            table_data = summary[table_name]
                            
                            # Check if it's a table-file reference (has 'path' key)
                            if isinstance(table_data, dict) and 'path' in table_data:
                                table_path = table_data['path']
                                table_file = run.file(table_path)
                                
                                # Download to temp directory
                                table_file.download(root=temp_dir, replace=True)
                                full_path = os.path.join(temp_dir, table_path)
                                
                                # Load and parse JSON directly
                                with open(full_path, 'r') as f:
                                    table_json = json.load(f)
                                
                                # Create DataFrame efficiently
                                if 'data' in table_json and 'columns' in table_json:
                                    df = pd.DataFrame(table_json['data'], columns=table_json['columns'])
                                    results[run_name]['tables'][table_name].append(df)
                                
                                # Remove file immediately after use
                                os.remove(full_path)
                                
                            # Handle wandb.Table object
                            elif hasattr(table_data, '_data'):
                                df = pd.DataFrame(table_data._data, columns=table_data._columns)
                                results[run_name]['tables'][table_name].append(df)
                            # Handle dict with data
                            elif isinstance(table_data, dict) and 'data' in table_data:
                                df = pd.DataFrame(table_data['data'], columns=table_data.get('columns', []))
                                results[run_name]['tables'][table_name].append(df)
                                    
                        except Exception as e:
                            print(f"Warning: Could not extract table '{table_name}' from run {run.name}: {e}")
                            continue
    finally:
        # Clean up temp directory
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def plot_calibration_curves(all_calibration_data: dict, output_dir: str, benchmark_title: str, baseline_methods, qlearning_methods, run_names):
    """
    Plot calibration curves for all run names and metrics.
    
    Args:
        all_calibration_data: Dictionary with run_name -> table_name -> list of DataFrames
        output_dir: Directory to save plots
        benchmark_title: Title of the benchmark (for plot titles)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Note: Color and linestyle are now determined by get_method_color_and_style() function
    
    for table_name in CALIBRATION_TABLES:
        # Determine metric type and split
        if 'ece' in table_name:
            metric_type = 'ECE'
        elif 'brier' in table_name:
            metric_type = 'Brier Score'
        else:
            metric_type = table_name.split('/')[-1]
        
        if 'val_unseen' in table_name:
            split = 'val_unseen'
        elif 'val_seen' in table_name:
            split = 'val_seen'
        else:
            split = 'unknown'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        for idx, run_name in enumerate(run_names):
            if run_name not in all_calibration_data:
                continue
            
            tables = all_calibration_data[run_name].get(table_name, [])
            if not tables:
                continue
            
            # Concatenate all dataframes efficiently (ignore_index is faster)
            combined_df = pd.concat(tables, ignore_index=True, copy=False)
            
            # Print columns for debugging
            print(f"  Processing {table_name} for {run_name}")
            print(f"    Columns: {combined_df.columns.tolist()}")
            print(f"    Shape: {combined_df.shape}")
            
            # Determine x and y columns in one pass
            x_col = next((col for col in ['quantile', 'timestep'] if col in combined_df.columns), None)
            y_col = next((col for col in ['brier_score', 'ece'] if col in combined_df.columns), None)
            
            if x_col is None or y_col is None:
                print(f"    Warning: Could not determine axes columns (x={x_col}, y={y_col})")
                continue
            
            print(f"    Using x={x_col}, y={y_col}")
            
            # Remove 'split' column early to reduce memory
            if 'split' in combined_df.columns:
                combined_df = combined_df.drop(columns=['split'])
            
            # Convert to numeric efficiently
            combined_df[x_col] = pd.to_numeric(combined_df[x_col], errors='coerce')
            combined_df[y_col] = pd.to_numeric(combined_df[y_col], errors='coerce')
            
            # Drop invalid rows
            combined_df = combined_df.dropna(subset=[x_col, y_col])
            
            if len(combined_df) == 0:
                print(f"    Warning: No valid data after conversion")
                continue
            
            print(f"    After conversion - x range: [{combined_df[x_col].min():.3f}, {combined_df[x_col].max():.3f}], y range: [{combined_df[y_col].min():.3f}, {combined_df[y_col].max():.3f}]")
            print(f"    Valid rows: {len(combined_df)}")
            
            # Group by x_col and calculate mean and std (optimized aggregation)
            try:
                grouped = combined_df.groupby(x_col, sort=False)[y_col].agg(['mean', 'std']).reset_index()
                
                # Print final quantile statistics for ECE and Brier unseen
                if ('ece' in table_name or 'brier' in table_name) and 'val_unseen' in table_name:
                    # Get last quantile value
                    last_quantile_row = grouped[grouped[x_col] == grouped[x_col].max()]
                    if not last_quantile_row.empty:
                        final_mean = last_quantile_row['mean'].values[0]
                        final_std = last_quantile_row['std'].values[0]
                        final_quantile = last_quantile_row[x_col].values[0]
                        print(f"    {run_name} - {metric_type} (val_unseen) at quantile {final_quantile:.2f}: {final_mean:.4f} ± {final_std:.4f}")
            except Exception as e:
                print(f"    Warning: Could not aggregate data: {e}")
                continue
                        
            # Plot with error bars
            label = get_method_label(run_name)
            color, linestyle = get_method_color_and_style(run_name, baseline_methods, qlearning_methods)

            ax.plot(grouped[x_col], grouped['mean'], label=label, linewidth=2.5, color=color, linestyle=linestyle)
            ax.fill_between(
                grouped[x_col],
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                alpha=0.2,
                color=color
            )
            has_data = True
        
        if has_data:
            ax.set_xlabel(x_col.replace('_', ' ').capitalize(), fontsize=12)
            ax.set_ylabel(metric_type, fontsize=12)
            ax.set_title(f'{benchmark_title}: {metric_type} - {split.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Set appropriate y-axis limits based on metric type
            if 'ECE' in metric_type:
                ax.set_ylim(bottom=0, top=0.5)
            elif 'Brier' in metric_type:
                ax.set_ylim(bottom=0, top=0.5)
            
            ax.set_xlim(left=-0.02, right=1)
            
            # Save plot
            plot_filename = f"{table_name.replace('/', '_')}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved calibration plot: {plot_path}")
        
        plt.close()
    
    print(f"\nAll calibration plots saved to: {output_dir}")


def save_calibration_data_to_csv(all_calibration_data: dict, output_dir: str, run_names: list):
    """
    Save all calibration curve data to CSV files.
    
    Args:
        all_calibration_data: Dictionary with run_name -> table_name -> list of DataFrames
        output_dir: Directory to save CSV files
        run_names: List of run names to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for table_name in CALIBRATION_TABLES:
        # Determine metric type and split
        if 'ece' in table_name:
            metric_type = 'ece'
        elif 'brier' in table_name:
            metric_type = 'brier'
        else:
            metric_type = table_name.split('/')[-1]
        
        if 'val_unseen' in table_name:
            split = 'val_unseen'
        elif 'val_seen' in table_name:
            split = 'val_seen'
        else:
            split = 'unknown'
        
        all_rows = []
        
        for run_name in run_names:
            if run_name not in all_calibration_data:
                continue
            
            tables = all_calibration_data[run_name].get(table_name, [])
            if not tables:
                continue
            
            # Concatenate all dataframes for this run
            combined_df = pd.concat(tables, ignore_index=True, copy=False)
            
            # Determine x and y columns
            x_col = next((col for col in ['quantile', 'timestep'] if col in combined_df.columns), None)
            y_col = next((col for col in ['brier_score', 'ece'] if col in combined_df.columns), None)
            
            if x_col is None or y_col is None:
                continue
            
            # Remove split column if present
            if 'split' in combined_df.columns:
                combined_df = combined_df.drop(columns=['split'])
            
            # Convert to numeric
            combined_df[x_col] = pd.to_numeric(combined_df[x_col], errors='coerce')
            combined_df[y_col] = pd.to_numeric(combined_df[y_col], errors='coerce')
            combined_df = combined_df.dropna(subset=[x_col, y_col])
            
            if len(combined_df) == 0:
                continue
            
            # Group by x_col and calculate statistics
            grouped = combined_df.groupby(x_col, sort=True)[y_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Add method information
            grouped['method'] = run_name
            grouped['method_label'] = get_method_label(run_name)
            grouped['metric_type'] = metric_type
            grouped['split'] = split
            
            # Rename columns for clarity
            grouped = grouped.rename(columns={
                x_col: 'time_quantile',
                'mean': f'{metric_type}_mean',
                'std': f'{metric_type}_std',
                'count': 'n_samples'
            })
            
            all_rows.append(grouped)
        
        if all_rows:
            # Combine all methods for this metric
            result_df = pd.concat(all_rows, ignore_index=True)
            
            # Reorder columns
            cols = ['method', 'method_label', 'metric_type', 'split', 'time_quantile', 
                    f'{metric_type}_mean', f'{metric_type}_std', 'n_samples']
            result_df = result_df[cols]
            
            # Sort by method and time_quantile
            result_df = result_df.sort_values(['method', 'time_quantile'])
            
            # Save to CSV
            csv_filename = f"{table_name.replace('/', '_')}_data.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            result_df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"Saved calibration data: {csv_path}")
    
    print(f"\nAll calibration data CSV files saved to: {output_dir}")


def plot_calibration_at_stop(calibration_at_stop_data: dict, output_dir: str, benchmark_title: str, baseline_methods, qlearning_methods, run_names):
    """
    Plot bar charts showing ECE and Brier scores at task_min_step - 1 (earliest stop).
    
    Args:
        calibration_at_stop_data: Dictionary with run_name -> metric_name -> list of values
        output_dir: Directory to save plots
        benchmark_title: Title of the benchmark (for plot titles)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Note: Color and linestyle are now determined by get_method_color_and_style() function
    
    # Group metrics by type (ECE/Brier) and split (seen/unseen)
    metric_groups = {
        'ece_val_seen': [],
        'ece_val_unseen': [],
        'brier_val_seen': [],
        'brier_val_unseen': []
    }
    
    for metric in CALIBRATION_AT_STOP_METRICS:
        if 'ece' in metric and 'val_seen' in metric:
            metric_groups['ece_val_seen'].append(metric)
        elif 'ece' in metric and 'val_unseen' in metric:
            metric_groups['ece_val_unseen'].append(metric)
        elif 'brier' in metric and 'val_seen' in metric:
            metric_groups['brier_val_seen'].append(metric)
        elif 'brier' in metric and 'val_unseen' in metric:
            metric_groups['brier_val_unseen'].append(metric)
    
    # Plot for each group
    for group_name, metrics in metric_groups.items():
        if not metrics:
            continue
        
        metric = metrics[0]  # Should only be one metric per group
        
        # Determine metric type and split
        if 'ece' in group_name:
            metric_type = 'ECE'
        elif 'brier' in group_name:
            metric_type = 'Brier Score'
        else:
            continue
        
        if 'val_unseen' in group_name:
            split = 'val_unseen'
        elif 'val_seen' in group_name:
            split = 'val_seen'
        else:
            continue
        
        # Collect data for all methods
        plot_data = []
        
        for run_name in run_names:
            if run_name not in calibration_at_stop_data:
                continue
            
            values = calibration_at_stop_data[run_name].get(metric, [])
            if not values:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            plot_data.append({
                'method': get_method_label(run_name),
                'method_full': run_name,
                'mean': mean_val,
                'std': std_val
            })
        
        if not plot_data:
            continue
        
        # Create bar plot
        df_plot = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        x_positions = np.arange(len(df_plot))
        colors = [get_method_color_and_style(row['method_full'], baseline_methods, qlearning_methods)[0] 
                  for _, row in df_plot.iterrows()]
        hatches = ['//' if get_method_color_and_style(row['method_full'], baseline_methods, qlearning_methods)[1] == '--' else '' 
                   for _, row in df_plot.iterrows()]
        
        bars = ax.bar(x_positions, df_plot['mean'], yerr=df_plot['std'],
                     color=colors, alpha=0.8, capsize=5, width=0.7)
        
        # Apply hatching pattern for TD-0 methods (dashed linestyle)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        # Customize plot
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel(metric_type, fontsize=12)
        ax.set_title(f'{metric_type} at task_min_step - {split.replace("_", " ").title()} ({benchmark_title})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df_plot['method'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0, top=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"calibration_at_stop_{group_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration at stop plot: {plot_path}")
        plt.close()
    
    print(f"\nCalibration at stop plots saved to: {output_dir}")


def plot_calibration_bars_at_selected_times(all_calibration_data: dict, output_dir: str, selected_times: list = None, benchmark_title: str = '', baseline_methods=None, qlearning_methods=None, run_names=None):
    """
    Plot bar charts showing calibration metrics at selected timesteps/quantiles.
    
    Args:
        all_calibration_data: Dictionary with run_name -> table_name -> list of DataFrames
        output_dir: Directory to save plots
        selected_times: List of timesteps/quantiles to show (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        benchmark_title: Title of the benchmark (for plot titles)
    """
    if selected_times is None:
        selected_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Note: Color and linestyle are now determined by get_method_color_and_style() function
    
    for table_name in CALIBRATION_TABLES:
        # Determine metric type and split
        if 'ece' in table_name:
            metric_type = 'ECE'
        elif 'brier' in table_name:
            metric_type = 'Brier Score'
        else:
            continue
        
        if 'val_unseen' in table_name:
            split = 'val_unseen'
        elif 'val_seen' in table_name:
            split = 'val_seen'
        else:
            continue
        
        # Collect data for all methods at selected times
        plot_data = []
        
        for run_name in run_names:
            if run_name not in all_calibration_data:
                continue
            
            tables = all_calibration_data[run_name].get(table_name, [])
            if not tables:
                continue
            
            combined_df = pd.concat(tables, ignore_index=True, copy=False)
            
            # Determine x and y columns
            x_col = next((col for col in ['quantile', 'timestep'] if col in combined_df.columns), None)
            y_col = next((col for col in ['brier_score', 'ece'] if col in combined_df.columns), None)
            
            if x_col is None or y_col is None:
                continue
            
            # Remove split column early
            if 'split' in combined_df.columns:
                combined_df = combined_df.drop(columns=['split'])
            
            # Convert to numeric efficiently
            combined_df[x_col] = pd.to_numeric(combined_df[x_col], errors='coerce')
            combined_df[y_col] = pd.to_numeric(combined_df[y_col], errors='coerce')
            combined_df = combined_df.dropna(subset=[x_col, y_col])
            
            if len(combined_df) == 0:
                continue
            
            # Group by x_col and calculate mean (don't sort for speed)
            grouped = combined_df.groupby(x_col, sort=False)[y_col].agg(['mean', 'std']).reset_index()
            
            # Extract values at selected times
            for time_point in selected_times:
                # Find closest time point
                closest_idx = (grouped[x_col] - time_point).abs().idxmin()
                closest_time = grouped.loc[closest_idx, x_col]
                
                # Only use if within 0.05 of target
                if abs(closest_time - time_point) < 0.05:
                    mean_val = grouped.loc[closest_idx, 'mean']
                    std_val = grouped.loc[closest_idx, 'std']
                    
                    plot_data.append({
                        'method': get_method_label(run_name),
                        'method_full': run_name,
                        'time': time_point,
                        'mean': mean_val,
                        'std': std_val
                    })
        
        if not plot_data:
            continue
        
        # Create bar plot
        df_plot = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Set up bar positions
        n_methods = len(run_names)
        n_times = len(selected_times)
        bar_width = 0.8 / n_methods
        
        for i, run_name in enumerate(run_names):
            method_data = df_plot[df_plot['method_full'] == run_name]
            if len(method_data) == 0:
                continue
            
            # Calculate x positions
            x_positions = np.arange(n_times) + i * bar_width
            
            # Filter to available times
            times_available = []
            means = []
            stds = []
            x_pos = []
            
            for j, time_point in enumerate(selected_times):
                time_data = method_data[method_data['time'] == time_point]
                if len(time_data) > 0:
                    times_available.append(time_point)
                    means.append(time_data['mean'].values[0])
                    stds.append(time_data['std'].values[0])
                    x_pos.append(x_positions[j])
            
            if means:
                label = get_method_label(run_name)
                color, linestyle = get_method_color_and_style(run_name, baseline_methods, qlearning_methods)
                hatch = '//' if linestyle == '--' else ''
                ax.bar(x_pos, means, bar_width, yerr=stds, 
                      label=label, color=color, alpha=0.8, capsize=3, hatch=hatch)
        
        # Customize plot
        ax.set_xlabel('Timestep / Quantile', fontsize=12)
        ax.set_ylabel(metric_type, fontsize=12)
        ax.set_title(f'{metric_type} at Selected Times - {split.replace("_", " ").title()} ({benchmark_title})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(n_times) + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels([f'{t:.1f}' for t in selected_times])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        if 'ECE' in metric_type:
            ax.set_ylim(bottom=0, top=0.5)
        elif 'Brier' in metric_type:
            ax.set_ylim(bottom=0, top=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{table_name.replace('/', '_')}_bars_selected_times.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar plot at selected times: {plot_path}")
        plt.close()
    
    print(f"\nBar plots at selected times saved to: {output_dir}")

def calculate_stats(values: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation."""
    if not values:
        return float('nan'), float('nan')
    return np.mean(values), np.std(values)

def extract_taskwise_metrics(project_name: str, run_name: str, num_tasks: int = 10) -> dict:
    """
    Extract taskwise ROC AUC metrics for all runs matching the given run name.
    
    Args:
        project_name: W&B project name
        run_name: Name/group to filter runs
        num_tasks: Number of tasks to extract metrics for
    
    Returns:
        Dictionary with structure: {task_id: {seed_id: {metric: value}}}
    """
    all_runs = get_runs(project_name)
    runs = filter_runs_by_name(all_runs, run_name)
    
    print(f"  Found {len(runs)} runs for {run_name}")
    
    # Structure: {task_id: {seed_id: {metric: value}}}
    results = defaultdict(lambda: defaultdict(dict))
    
    # Pre-generate metric names for efficiency
    metric_names = [
        (task_id, split, f"falert_early_roc_auc_taskwise/model_val_{split}_{task_id}")
        for task_id in range(num_tasks)
        for split in ['seen', 'unseen']
    ]
    
    for run in runs:
        # Extract seed number from config or name
        try:
            seed = run.config.get('train', {}).get('seed', 0)
        except:
            suffix = run.name[len(run_name):]
            seed = int(suffix[1:].split('-')[0]) if suffix and suffix.startswith('-') else 0
        
        summary = run.summary._json_dict
        
        # Batch extract metrics
        for task_id, split, metric_name in metric_names:
            if metric_name in summary:
                results[task_id][seed][f"val_{split}"] = summary[metric_name]
    
    return dict(results)


def compare_groups_per_task(baseline_data: dict, qlearning_data: dict, task_id: int, metric: str) -> dict:
    """
    Compare baseline vs q-learning methods for a specific task and metric.
    
    Args:
        baseline_data: Dictionary {method_name: {task_id: {seed_id: {metric: value}}}}
        qlearning_data: Dictionary {method_name: {task_id: {seed_id: {metric: value}}}}
        task_id: Task ID to compare
        metric: Metric name (e.g., 'val_unseen')
    
    Returns:
        Dictionary with comparison statistics
    """
    # Collect all values from all methods in each group
    baseline_values = []
    qlearning_values = []
    
    for method_data in baseline_data.values():
        if task_id in method_data:
            for seed_data in method_data[task_id].values():
                if metric in seed_data:
                    baseline_values.append(seed_data[metric])
    
    for method_data in qlearning_data.values():
        if task_id in method_data:
            for seed_data in method_data[task_id].values():
                if metric in seed_data:
                    qlearning_values.append(seed_data[metric])
    
    if not baseline_values or not qlearning_values:
        return None
    
    # Calculate statistics
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    qlearning_mean = np.mean(qlearning_values)
    qlearning_std = np.std(qlearning_values)
    
    # Statistical tests
    t_stat, t_pval = stats.ttest_ind(baseline_values, qlearning_values)
    u_stat, u_pval = stats.mannwhitneyu(baseline_values, qlearning_values, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_std**2 + 
                          (len(qlearning_values) - 1) * qlearning_std**2) / 
                         (len(baseline_values) + len(qlearning_values) - 2))
    cohens_d = (qlearning_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        'task_id': task_id,
        'metric': metric,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'baseline_n': len(baseline_values),
        'qlearning_mean': qlearning_mean,
        'qlearning_std': qlearning_std,
        'qlearning_n': len(qlearning_values),
        'mean_diff': qlearning_mean - baseline_mean,
        't_test_p': t_pval,
        'mann_whitney_p': u_pval,
        'cohens_d': cohens_d,
        'winner': 'Q-Learning' if qlearning_mean > baseline_mean else 'Baseline',
        'significant': t_pval < 0.05
    }


def plot_taskwise_comparison(comparison_results: list, metric: str, output_dir: str):
    """
    Plot comparison across all tasks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(comparison_results)
    if len(df) == 0:
        print(f"No data to plot for {metric}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean comparison
    x = df['task_id']
    baseline_means = df['baseline_mean']
    qlearning_means = df['qlearning_mean']
    baseline_stds = df['baseline_std']
    qlearning_stds = df['qlearning_std']
    
    ax1.errorbar(x, baseline_means, yerr=baseline_stds, 
                 label='Baseline Methods', marker='o', linewidth=2, capsize=5)
    ax1.errorbar(x, qlearning_means, yerr=qlearning_stds, 
                 label='Q-Learning Methods', marker='s', linewidth=2, capsize=5)
    
    ax1.set_xlabel('Task ID', fontsize=12)
    ax1.set_ylabel('ROC AUC', fontsize=12)
    ax1.set_title(f'Taskwise Performance - {metric}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference (Q-Learning - Baseline)
    differences = df['mean_diff']
    colors = ['green' if d > 0 else 'red' for d in differences]
    significant = df['significant']
    
    bars = ax2.bar(x, differences, color=colors, alpha=0.6)
    
    # Mark significant differences with asterisks
    for i, (task, diff, sig) in enumerate(zip(x, differences, significant)):
        if sig:
            ax2.text(task, diff, '*', ha='center', va='bottom' if diff > 0 else 'top', 
                    fontsize=16, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Task ID', fontsize=12)
    ax2.set_ylabel('Difference (Q-Learning - Baseline)', fontsize=12)
    ax2.set_title(f'Performance Difference - {metric}\n(* = p < 0.05)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'taskwise_comparison_{metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()


def compare_methods_per_task_pairwise(all_methods_data: dict, task_id: int, metric: str) -> list:
    """
    Compare all methods pairwise for a specific task and metric.
    
    Args:
        all_methods_data: Dictionary {method_name: {task_id: {seed_id: {metric: value}}}}
        task_id: Task ID to compare
        metric: Metric name (e.g., 'val_unseen')
    
    Returns:
        List of dictionaries with pairwise comparison statistics
    """
    results = []
    method_names = list(all_methods_data.keys())
    
    # Pre-extract values for all methods to avoid redundant loops
    method_values = {}
    for method in method_names:
        values = []
        if task_id in all_methods_data[method]:
            for seed_data in all_methods_data[method][task_id].values():
                if metric in seed_data:
                    try:
                        val = float(seed_data[metric])
                        values.append(val)
                    except (ValueError, TypeError):
                        continue
        if values:
            method_values[method] = np.array(values, dtype=float)  # Ensure float dtype
    
    # Perform pairwise comparisons
    for method1, method2 in combinations(method_names, 2):
        if method1 not in method_values or method2 not in method_values:
            continue
        
        values1 = method_values[method1]
        values2 = method_values[method2]
        
        # Calculate statistics using numpy (more efficient)
        mean1, std1 = values1.mean(), values1.std()
        mean2, std2 = values2.mean(), values2.std()
        
        # Statistical tests
        t_stat, t_pval = stats.ttest_ind(values1, values2)
        u_stat, u_pval = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + 
                              (len(values2) - 1) * std2**2) / 
                             (len(values1) + len(values2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'task_id': task_id,
            'method_1': method1.replace('openvla-10-', ''),
            'method_2': method2.replace('openvla-10-', ''),
            'mean_1': mean1,
            'std_1': std1,
            'n_1': len(values1),
            'mean_2': mean2,
            'std_2': std2,
            'n_2': len(values2),
            'mean_diff': mean1 - mean2,
            't_test_p': t_pval,
            'mann_whitney_p': u_pval,
            'cohens_d': cohens_d,
            'winner': method1 if mean1 > mean2 else method2,
            'significant': t_pval < 0.05
        })
    
    return results


def create_taskwise_summary_table(comparison_results: list) -> pd.DataFrame:
    """
    Create a summary table with win/loss statistics for taskwise comparison.
    """
    df = pd.DataFrame(comparison_results)
    
    summary = {
        'Total Tasks': len(df),
        'Q-Learning Wins': (df['winner'] == 'Q-Learning').sum(),
        'Baseline Wins': (df['winner'] == 'Baseline').sum(),
        'Significant Differences': df['significant'].sum(),
        'Avg Difference': df['mean_diff'].mean(),
        'Median Difference': df['mean_diff'].median(),
        'Q-Learning Win %': (df['winner'] == 'Q-Learning').sum() / len(df) * 100,
    }
    
    return pd.DataFrame([summary])


def main(args: argparse.Namespace):
    print(f"Extracting metrics from project: {PROJECT_NAME}")
    # Use run_names from get_methods_for_benchmark
    baseline_methods, qlearning_methods, run_names, benchmark_title = get_methods_for_benchmark(args.benchmark)
    print(f"Run names: {run_names}")
    if args.benchmark.lower() == "droid":
        metrics = METRICS_DROID
    else:
        metrics = METRICS
    print(f"Metrics: {metrics}\n")
    
    # Try to load existing results to skip redundant W&B API calls
    output_path = args.output + "_" + args.benchmark + ".csv"
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_run_names = set(existing_df['run_name'].values)
        print(f"Loaded {len(existing_run_names)} cached runs from {output_path}")
    else:
        existing_df = None
        existing_run_names = set()

    all_results = [] if existing_df is None else existing_df.to_dict('records')
    all_calibration_data = {}
    all_calibration_at_stop = {}
    all_raw_metrics = {metric: {} for metric in metrics}  # Store raw values for statistical tests

    # Extract data for all run names in one pass
    print("Extracting data for all run names in a single pass...")
    if args.plot_calibration:
        all_run_data = extract_run_data(
            PROJECT_NAME,
            run_names,
            metrics=metrics + CALIBRATION_AT_STOP_METRICS,
            table_names=CALIBRATION_TABLES
        )
    else:
        all_run_data = extract_run_data(PROJECT_NAME, run_names, metrics=metrics)
    
    # Process each run name
    for run_name in run_names:
        print(f"Processing: {run_name}")
        
        run_data = all_run_data[run_name]
        
        # Split the metrics
        metrics_data = {k: v for k, v in run_data['metrics'].items() if k in metrics}
        
        if args.plot_calibration:
            at_stop_data = {k: v for k, v in run_data['metrics'].items() if k in CALIBRATION_AT_STOP_METRICS}
            all_calibration_data[run_name] = run_data['tables']
            all_calibration_at_stop[run_name] = at_stop_data
            
            for table_name, tables in run_data['tables'].items():
                print(f"  {table_name}: {len(tables)} runs")
            
            for metric_name, values in at_stop_data.items():
                if values:
                    metric_short = metric_name.split('/')[-1]
                    print(f"  {metric_short}: {np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})")

        # Calculate statistics for each metric
        row = {"run_name": run_name}

        for metric in metrics:
            values = metrics_data[metric]
            mean_val, std_val = calculate_stats(values)

            # Store raw values for statistical tests
            all_raw_metrics[metric][run_name] = values

            # Shorten metric name for display
            metric_short = metric.split('/')[-1]

            row[f"{metric_short}_mean"] = mean_val
            row[f"{metric_short}_std"] = std_val
            row[f"{metric_short}_n"] = len(values)

            print(f"  {metric_short}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")
            print(f"    Values: {values}")

        all_results.append(row)
        print()
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    cols = ["run_name"]
    for metric in metrics:
        metric_short = metric.split('/')[-1]
        cols.extend([f"{metric_short}_mean", f"{metric_short}_std", f"{metric_short}_n"])
    df = pd.DataFrame(all_results)
    df = df[cols]
    
    # Perform statistical tests
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS")
    print("=" * 100)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved ROC AUC results to: {output_path}")
    
    return df, all_calibration_data, all_calibration_at_stop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ROC AUC results and calibration curves for specific run names from W&B"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./scripts/roc_auc_results",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--plot-calibration",
        action="store_true",
        help="Plot calibration curves",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="./scripts/calibration_plots_selected_times",
        help="Directory to save calibration plots",
    )
    parser.add_argument(
        "--taskwise",
        action="store_true",
        help="Perform taskwise comparison between Q-learning and baseline methods",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks to analyze for taskwise comparison",
    )
    parser.add_argument(
        "--plot-taskwise",
        action="store_true",
        help="Generate taskwise comparison plots",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero",
        choices=["libero", "widowx", "droid", "libero_fast_pi0", "libero_pi0", "univla"],
        help="Benchmark to analyze: libero, widowx, droid, or libero_pi0",
    )
    args = parser.parse_args()
    args.plot_calibration = True  # Always plot calibration curves
    args.taskwise = True        # Always perform taskwise comparison

    PROJECT_NAME = set_project_name(args.benchmark)
    baseline_methods, qlearning_methods, run_names, benchmark_title = get_methods_for_benchmark(args.benchmark)
    df, all_calibration_data, all_calibration_at_stop = main(args)
    if args.plot_calibration and all_calibration_data:
        # print("\n" + "=" * 100)
        # print("PLOTTING CALIBRATION CURVES")
        # print("=" * 100)
        plot_dir = args.plot_dir
        plot_calibration_curves(all_calibration_data, plot_dir, benchmark_title, baseline_methods, qlearning_methods, run_names)
        # print("=" * 100)
        
        # Save calibration data to CSV
        print("\n" + "=" * 100)
        print("SAVING CALIBRATION DATA TO CSV")
        print("=" * 100)
        save_calibration_data_to_csv(all_calibration_data, plot_dir, run_names)
        print("=" * 100)
        
    if args.plot_calibration and all_calibration_at_stop:
        print("\n" + "=" * 100)
        print("PLOTTING CALIBRATION AT SELECTED TIMES")
        plot_calibration_bars_at_selected_times(all_calibration_data, plot_dir, selected_times=None, benchmark_title=benchmark_title, baseline_methods=baseline_methods, qlearning_methods=qlearning_methods, run_names=run_names)
        print("\n" + "=" * 100)
        print("PLOTTING CALIBRATION AT STOP (task_min_step - 1)")
        print("=" * 100)
        plot_calibration_at_stop(all_calibration_at_stop, plot_dir, benchmark_title, baseline_methods, qlearning_methods, run_names)
        print("=" * 100)
