"""
Create unified Brier score val_unseen plot with 4 subplots (one per benchmark).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 13

# File paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO_ROOT)
LIBERO_BRIER = os.path.join(DATA_DIR, "plots_libero/calibration_curves_model_brier_val_unseen_table_data.csv")
WIDOWX_BRIER = os.path.join(DATA_DIR, "plots_widowx/calibration_curves_model_brier_val_unseen_table_data.csv")
DROID_BRIER = os.path.join(DATA_DIR, "plots_droid/calibration_curves_model_brier_val_unseen_table_data.csv")
PI0_BRIER = os.path.join(DATA_DIR, "plots_pi0_libero/calibration_curves_model_brier_val_unseen_table_data.csv")

# Color and style mappings (from your script)
COLOR_GROUPS = {
    "mlp": "#2E86AB",
    "lstm": "#94C05B",
    "qlearning": "#820263",
}

LABEL_MAPPING = {
    # LIBERO
    "openvla-10-indep-mlp_MSELoss": "MLP (BCE Loss)",
    "openvla-10-indep-mlp_TDLoss_best_brier": "MLP (TD-0)",
    "openvla-10-lstm-lstm_clean": "LSTM (BCE Loss)",
    "openvla-10-lstm-lstm_TDLoss_best_brier": "LSTM (TD-0)",
    "openvla-10-q_learning-top_k_probs_BCE_best_roc": "TDQC (Top-10 BCE Loss)",
    "openvla-10-q_learning-top_k_probs": "TDQC (Top-10 TD-0)",
    
    # WidowX
    "openvla-widowx-indep-mlp_MSELoss_best": "MLP (BCE Loss)",
    "openvla-widowx-indep-mlp_TDLoss_best_new": "MLP (TD-0)",
    "openvla-widowx-lstm-lstm_best": "LSTM (BCE Loss)",
    "openvla-widowx-lstm-lstm_TD0_best_new": "LSTM (TD-0)",
    "openvla-widowx-q_learning-top_k_probs_BCE_best": "TDQC (Top-10 BCE Loss)",
    "openvla-widowx-q_learning-widowx_top_k_probs_TD0_best": "TDQC (Top-10 TD-0)",
    
    # Droid
    "pizero_fast_droid-0510-indep-mlp_best_5090_BCE_best_roc": "MLP (BCE Loss)",
    "pizero_fast_droid-0510-indep-mlp_5090_TDLoss_best_brier": "MLP (TD-0)",
    "pizero_fast_droid-0510-lstm-lstm_best_5090": "LSTM (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_5090_TDLoss_best_brier": "LSTM (TD-0)",
    
    # Pi0 LIBERO
    "pizero-default-indep-mlp_BCE_best_roc": "MLP (BCE Loss)",
    "pizero-default-indep-mlp_TD0_best_brier": "MLP (TD-0)",
    "pizero-default-lstm-lstm_best": "LSTM (BCE Loss)",
    "pizero-default-lstm-lstm_TD0_best_brier": "LSTM (TD-0)",
}

def get_method_color_and_style(method_name: str) -> tuple:
    """Get color and linestyle for a method."""
    # Determine method type
    if 'mlp' in method_name.lower() or 'indep' in method_name.lower():
        color = COLOR_GROUPS['mlp']
    elif 'lstm' in method_name.lower():
        color = COLOR_GROUPS['lstm']
    elif 'q_learning' in method_name.lower() or 'qlearning' in method_name.lower():
        color = COLOR_GROUPS['qlearning']
    else:
        color = '#999999'
    
    # Determine linestyle (BCE Loss = solid, TD-0 = dashed)
    if 'BCE' in method_name or 'MSELoss' in method_name or 'clean' in method_name or method_name.endswith('_best') or method_name.endswith('_best_5090'):
        linestyle = '-'
    else:
        linestyle = '--'
    
    return color, linestyle

def plot_benchmark_subplot(ax, csv_path, title):
    """Plot a single benchmark subplot as grouped bar chart by time quantile."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping {title}")
        ax.text(0.5, 0.5, f"{title}\n(Data not available)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Filter for val_unseen
    df_val = df[df['split'] == 'val_unseen'].copy()
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No val_unseen data)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Filter for specific time quantiles only
    target_quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df_val = df_val[df_val['time_quantile'].isin(target_quantiles)].copy()
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No data for target quantiles)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Get labels and colors
    df_val['label'] = df_val['method'].map(LABEL_MAPPING)
    df_val = df_val.dropna(subset=['label'])  # Remove methods not in mapping
    df_val['color'] = df_val['method'].apply(lambda x: get_method_color_and_style(x)[0])
    df_val['is_td0'] = df_val['label'].str.contains('TD-0', na=False)
    
    if df_val.empty:
        ax.text(0.5, 0.5, f"{title}\n(No mapped methods)", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Get unique methods in order
    method_order = []
    for base in ["MLP", "LSTM", "TDQC"]:
        for loss in ["BCE Loss", "TD-0"]:
            pattern = f"{base} \\({loss}\\)" if base != "TDQC" else f"{base} \\(Top-10 {loss}\\)"
            matching = df_val[df_val['label'].str.contains(pattern, na=False, regex=True)]['label'].unique()
            if len(matching) > 0:
                method_order.extend(matching.tolist())
    
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
        is_td0 = method_data.iloc[0]['is_td0']
        hatch = '///' if is_td0 else None
        
        bars = ax.bar(x_pos + offset,
                     method_data['brier_mean'],
                     bar_width,
                     yerr=method_data['brier_std'],
                     label=method_label,
                     color=color,
                     alpha=0.8,
                     capsize=3,
                     hatch=hatch,
                     edgecolor='white' if is_td0 else None,
                     linewidth=1.5 if is_td0 else 0,
                     error_kw={'linewidth': 1.0})
        
        legend_handles[method_label] = bars
    
    # Customize subplot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q:.1f}' for q in time_quantiles], fontsize=12)
    ax.set_xlabel('Time Quantile', fontsize=13)
    ax.set_ylabel('Brier Score', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    max_val = (df_val['brier_mean'] + df_val['brier_std']).max()
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.15)
    
    return legend_handles

def main():
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    axes = axes.flatten()
    
    # Plot each benchmark
    benchmarks = [
        (LIBERO_BRIER, "OpenVLA-LIBERO", axes[0]),
        (WIDOWX_BRIER, "OpenVLA-WidowX", axes[1]),
        (DROID_BRIER, "Pi0-FAST-Droid", axes[2]),
        (PI0_BRIER, "Pi0-LIBERO", axes[3])
    ]
    
    all_handles = {}
    
    for csv_path, title, ax in benchmarks:
        handles = plot_benchmark_subplot(ax, csv_path, title)
        all_handles.update(handles)
    
    # Create single legend for all subplots
    # Sort legend entries: MLP, LSTM, TDQC, with BCE before TD-0
    legend_order = [
        "MLP (BCE Loss)", "MLP (TD-0)",
        "LSTM (BCE Loss)", "LSTM (TD-0)",
        "TDQC (Top-10 BCE Loss)", "TDQC (Top-10 TD-0)"
    ]
    
    ordered_handles = []
    ordered_labels = []
    for label in legend_order:
        if label in all_handles:
            ordered_handles.append(all_handles[label])
            ordered_labels.append(label)
    
    # Add legend below the subplots
    fig.legend(ordered_handles, ordered_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=3, 
              fontsize=12,
              frameon=True)
    
    # Add overall title
    fig.suptitle('Brier Score on Validation Unseen Set Across Benchmarks', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot
    output_path = os.path.join(DATA_DIR, "unified_brier_val_unseen.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved unified plot to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
