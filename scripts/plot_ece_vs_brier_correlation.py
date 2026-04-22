"""
Create correlation plot between ECE and Brier scores across all benchmarks.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from pathlib import Path

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color mappings
COLOR_GROUPS = {
    "mlp": "#2E86AB",
    "lstm": "#94C05B",
    "qlearning": "#820263",
}

REPO_ROOT = Path(__file__).resolve().parents[1]

def get_method_color(method_name, is_td0=False):
    """Get color for a method, with lighter shade for BCE."""
    if 'mlp' in method_name.lower() or 'indep' in method_name.lower():
        base_color = COLOR_GROUPS['mlp']
    elif 'q_learning' in method_name.lower() or 'qlearning' in method_name.lower() or 'top_k_probs' in method_name.lower() or 'Top-10' in method_name.lower():
        base_color = COLOR_GROUPS['qlearning']
    elif 'lstm' in method_name.lower():
        base_color = COLOR_GROUPS['lstm']
    else:
        base_color = '#999999'
    
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    # Lighten for BCE (non-TD0)
    if not is_td0:
        r = min(1.0, r + 0.4 * (1.0 - r))
        g = min(1.0, g + 0.4 * (1.0 - g))
        b = min(1.0, b + 0.4 * (1.0 - b))
    
    return (r, g, b)

def is_td0_method(method_name):
    """Determine if a method uses TD-0 loss."""
    if 'BCE' in method_name or 'MSELoss' in method_name or 'clean' in method_name or 'Qiao' in method_name:
        return False
    # Special case: q_learning methods without BCE are TD-0
    if 'q_learning' in method_name and 'BCE' not in method_name or 'top_k_probs' in method_name.lower() or 'Top-10' in method_name.lower():
        return True
    if 'TD' in method_name or 'TDLoss' in method_name or 'TD0' in method_name or 'td_lambda' in method_name:
        return True

    return False

def load_correlation_data():
    """Load ECE and Brier data from all benchmarks."""
    
    # Benchmark folders
    benchmark_folders = [
        ('plots_libero', 'OpenVLA-LIBERO'),
        ('plots_widowx', 'OpenVLA-WidowX'),
        ('plots_droid', 'π₀-FAST-Droid'),
        ('plots_pi0_libero', 'π₀-LIBERO'),
        ('plots_pi0_fast_libero', 'π₀-FAST-LIBERO'),
        ('plots_univla', 'UniVLA-LIBERO'),
    ]
    
    all_data = []
    
    for folder, benchmark_name in benchmark_folders:
        base_path = REPO_ROOT / folder
        
        # Load both splits
        for split in ['val_seen', 'val_unseen']:
            ece_file = base_path / f"calibration_curves_model_ece_{split}_table_data.csv"
            brier_file = base_path / f"calibration_curves_model_brier_{split}_table_data.csv"
            
            if not os.path.exists(ece_file) or not os.path.exists(brier_file):
                print(f"Warning: Missing files for {benchmark_name} {split}")
                continue
            
            # Load data
            ece_df = pd.read_csv(ece_file)
            brier_df = pd.read_csv(brier_file)
            
            # Filter for specific time quantiles
            target_quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ece_df = ece_df[ece_df['time_quantile'].isin(target_quantiles)]
            brier_df = brier_df[brier_df['time_quantile'].isin(target_quantiles)]
            
            # Merge on method and time_quantile
            merged = pd.merge(
                ece_df[['method', 'time_quantile', 'ece_mean']],
                brier_df[['method', 'time_quantile', 'brier_mean']],
                on=['method', 'time_quantile'],
                how='inner'
            )
            
            merged['benchmark'] = benchmark_name
            merged['split'] = split
            merged['is_td0'] = merged['method'].apply(is_td0_method)
            
            all_data.append(merged)
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    return df

def plot_ece_vs_brier():
    """Create combined scatter plot of ECE vs Brier scores for all benchmarks."""
    
    # Load data
    df = load_correlation_data()
    
    print(f"Loaded {len(df)} data points from {df['benchmark'].nunique()} benchmarks")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Splits: {df['split'].unique()}")
    
    benchmarks = ['OpenVLA-LIBERO', 'OpenVLA-WidowX', 'π₀-FAST-Droid', 'π₀-LIBERO', 'π₀-FAST-LIBERO', 'UniVLA-LIBERO']
    
    # Create figure with 6 rows x 2 columns (one row per benchmark, columns for seen/unseen)
    fig, axes = plt.subplots(6, 2, figsize=(16, 26))
    
    for bench_idx, benchmark in enumerate(benchmarks):
        benchmark_df = df[df['benchmark'] == benchmark]
        
        for split_idx, split in enumerate(['val_seen', 'val_unseen']):
            ax = axes[bench_idx, split_idx]
            split_df = benchmark_df[benchmark_df['split'] == split]
            
            if len(split_df) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot points
            for _, row in split_df.iterrows():
                color = get_method_color(row['method'], row['is_td0'])
                marker = 'o' if split == 'val_unseen' else '^'
                
                ax.scatter(row['brier_mean'], row['ece_mean'], 
                          c=[color], s=100, alpha=0.7, 
                          marker=marker, edgecolors='black', linewidths=0.8)
            
            # Calculate correlation
            if len(split_df) > 2:
                spearman_corr, spearman_p = stats.spearmanr(split_df['brier_mean'], split_df['ece_mean'])
                
                # Fit linear regression
                z = np.polyfit(split_df['brier_mean'], split_df['ece_mean'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(split_df['brier_mean'].min(), split_df['brier_mean'].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2)
                
                # Add correlation text (smaller font for combined plot)
                corr_text = f"ρ = {spearman_corr:.3f}"
                
                ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Labels and styling
            ax.set_xlabel('Brier Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('ECE Score', fontsize=11, fontweight='bold')
            split_label = 'Seen' if split == 'val_seen' else 'Unseen'
            ax.set_title(f'{benchmark} - {split_label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Create custom legend for method types
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('mlp', False),
             markersize=10, label='MLP feat (BCE)', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('mlp', True),
             markersize=10, label='MLP feat (TDQC)', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('lstm', False),
             markersize=10, label='RNN feat (BCE)', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('lstm', True),
             markersize=10, label='RNN feat (TDQC)', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('qlearning', False),
             markersize=10, label='RNN top-10 (BCE)', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_method_color('qlearning', True),
             markersize=10, label='RNN top-10 (TDQC)', markeredgecolor='black', markeredgewidth=0.8),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    # plt.suptitle('ECE vs Brier Score Correlation - All Benchmarks', 
    #             fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # Save combined figure
    output_path = Path(__file__).resolve().with_name("ece_vs_brier_all_benchmarks.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")
    plt.close()
    
    # Save data to CSV
    csv_path = Path(__file__).resolve().with_name("ece_vs_brier_correlation_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    return df

if __name__ == '__main__':
    df = plot_ece_vs_brier()
    print("\nData summary:")
    print(f"Total data points: {len(df)}")
    print(f"\nBy split:")
    print(df.groupby('split').size())
    print(f"\nBy benchmark:")
    print(df.groupby('benchmark').size())
