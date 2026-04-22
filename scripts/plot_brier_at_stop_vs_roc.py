"""
Create scatter plot of Brier Score at stop time vs ROC-AUC across all methods and benchmarks.
Uses Brier scores at stop time (T_hat) from the calibration tables.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO_ROOT)

# ROC-AUC CSV files
ROC_FILES = {
    'OpenVLA-LIBERO':       'scripts/plots_libero_libero.csv',
    'OpenVLA-WidowX':       'plots_widowx/plots_widowx_widowx.csv',
    r'$\pi_0$-FAST-Droid':  'scripts/roc_auc_results_pi0_fast_droid.csv',
    r'$\pi_0$-LIBERO':      'scripts/roc_auc_results_libero_pi0.csv',
    r'$\pi_0$-FAST-LIBERO': 'scripts/roc_auc_results_libero_fast_pi0.csv',
    'UniVLA-LIBERO':        'scripts/roc_auc_results_univla.csv',
}

# Brier scores at stop time from the calibration curve CSVs (last time quantile)
# Format: (benchmark, method, seen_mean, seen_std, unseen_mean, unseen_std)
BRIER_AT_STOP = [
    # OpenVLA-LIBERO
    ('OpenVLA-LIBERO', 'openvla-10-indep-mlp_BCE',             0.1920, 0.0200, 0.2310, 0.0200),
    ('OpenVLA-LIBERO', 'openvla-10-indep-mlp_TD0',             0.1950, 0.0220, 0.2290, 0.0220),
    ('OpenVLA-LIBERO', 'openvla-10-lstm-lstm',                 0.2040, 0.0380, 0.2550, 0.0590),
    ('OpenVLA-LIBERO', 'openvla-10-lstm-lstm_TD0',             0.1970, 0.0240, 0.2180, 0.0200),
    ('OpenVLA-LIBERO', 'openvla-10-q_learning-top_k_probs_BCE',0.1990, 0.0160, 0.2060, 0.0220),
    ('OpenVLA-LIBERO', 'openvla-10-q_learning-top_k_probs_TD0',0.1910, 0.0150, 0.1970, 0.0210),

    # OpenVLA-WidowX
    ('OpenVLA-WidowX', 'openvla-widowx-indep-mlp_BCE',             0.1277, 0.0197, 0.1648, 0.0350),
    ('OpenVLA-WidowX', 'openvla-widowx-indep-mlp_TD0',             0.1300, 0.0226, 0.1693, 0.0399),
    ('OpenVLA-WidowX', 'openvla-widowx-lstm-lstm',                 0.1692, 0.0629, 0.2138, 0.0770),
    ('OpenVLA-WidowX', 'openvla-widowx-lstm-lstm_TD0',             0.0961, 0.0220, 0.1531, 0.0582),
    ('OpenVLA-WidowX', 'openvla-widowx-q_learning-top_k_probs_BCE',0.3013, 0.0635, 0.3441, 0.0493),
    ('OpenVLA-WidowX', 'openvla-widowx-q_learning-top_k_probs',    0.1564, 0.0238, 0.1923, 0.0399),

    # Pi0-FAST-LIBERO
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-indep-mlp_BCE',                 0.1030, 0.0190, 0.1620, 0.0530),
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-indep-mlp_TD0',                 0.1090, 0.0220, 0.1500, 0.0350),
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-lstm-lstm',                     0.1060, 0.0180, 0.1480, 0.0420),
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-lstm-lstm_TD0',                 0.1030, 0.0280, 0.1630, 0.0600),
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-lstm-lstm_top_k_probs_BCE',     0.1219, 0.0221, 0.1581, 0.0532),
    (r'$\pi_0$-FAST-LIBERO', 'pizero_fast-default-lstm-lstm_top_k_probs_TD0',     0.1053, 0.0203, 0.1412, 0.0448),

    # Pi0-FAST-Droid
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-indep-mlp_BCE',     0.2068, 0.0172, 0.2483, 0.0241),
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-indep-mlp_TD0',     0.2101, 0.0085, 0.2296, 0.0137),
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-lstm-lstm',         0.2208, 0.0485, 0.2886, 0.0571),
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-lstm-lstm_TD0',     0.1504, 0.0271, 0.2154, 0.0394),
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE', 0.2379, 0.0058, 0.2434, 0.0054),
    (r'$\pi_0$-FAST-Droid', 'pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0', 0.2046, 0.0153, 0.2281, 0.0222),

    # Pi0-LIBERO
    (r'$\pi_0$-LIBERO', 'pizero-default-indep-mlp_BCE',     0.0750, 0.0180, 0.1370, 0.0580),
    (r'$\pi_0$-LIBERO', 'pizero-default-indep-mlp_TD0',     0.0680, 0.0210, 0.1280, 0.0600),
    (r'$\pi_0$-LIBERO', 'pizero-default-lstm-lstm',         0.1230, 0.0430, 0.1720, 0.0950),
    (r'$\pi_0$-LIBERO', 'pizero-default-lstm-lstm_TD0',     0.0610, 0.0190, 0.0970, 0.0330),

    # UniVLA-LIBERO
    ('UniVLA-LIBERO', 'univla-default-indep-mlp_BCE',                   0.0914, 0.0220, 0.1576, 0.0358),
    ('UniVLA-LIBERO', 'univla-default-indep-mlp_TD0',                   0.0661, 0.0151, 0.1307, 0.0275),
    ('UniVLA-LIBERO', 'univla-default-lstm-lstm_BCE',                   0.1240, 0.0198, 0.1619, 0.0144),
    ('UniVLA-LIBERO', 'univla-default-lstm-lstm_TD0',                   0.0643, 0.0156, 0.0996, 0.0262),
    ('UniVLA-LIBERO', 'univla-default-lstm-lstm_top_k_probs_BCE',       0.1381, 0.0468, 0.1524, 0.0645),
    ('UniVLA-LIBERO', 'univla-default-lstm-lstm_top_k_probs_TD0',       0.0996, 0.0246, 0.1071, 0.0560),
]

# Color mapping
COLOR_GROUPS = {
    "mlp": "#2E86AB",
    "lstm": "#94C05B",
    "qlearning": "#820263",
}

def get_method_color(method_name: str, is_td0: bool) -> str:
    """Get color for a method, with lighter shade for BCE methods."""
    # Determine base color by method type
    if 'mlp' in method_name.lower() or 'indep' in method_name.lower():
        base_color = COLOR_GROUPS['mlp']
    elif 'top_k_probs' in method_name.lower() or 'q_learning' in method_name.lower() or 'qlearning' in method_name.lower():
        base_color = COLOR_GROUPS['qlearning']
    elif 'lstm' in method_name.lower():
        base_color = COLOR_GROUPS['lstm']
    else:
        base_color = '#999999'
    
    # If BCE (not TD-0), return lighter version
    if not is_td0:
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(base_color)
        # Lighten by mixing with white (increase each channel by 40%)
        lighter_rgb = tuple(min(1.0, c + 0.4 * (1.0 - c)) for c in rgb)
        return mcolors.to_hex(lighter_rgb)
    
    return base_color

def is_td0_method(method_name: str) -> bool:
    """Check if method uses TD-0 loss."""
    # List of BCE/non-TD indicators
    bce_indicators = ['BCE', 'MSELoss', 'clean', 'Qiao']
    
    # If it has BCE indicators, it's not TD-0
    if any(bce in method_name for bce in bce_indicators):
        return False
    
    # TD indicators
    td_indicators = ['TD', 'TDLoss', 'TD0', 'td_lambda']
    
    # If it has TD indicators, it's TD-0
    if any(td in method_name for td in td_indicators):
        return True
    
    # Special case: top-k probability methods without explicit BCE/TD marker
    # If it's a q_learning method and doesn't have BCE, it's TD-0
    if 'q_learning' in method_name.lower() or 'qlearning' in method_name.lower():
        return True
    
    return False

def load_data():
    """Load and merge ROC-AUC and Brier score data."""
    data_points = []
    
    # Create a dataframe from Brier at stop data
    brier_df = pd.DataFrame(BRIER_AT_STOP, 
                            columns=['benchmark', 'method', 'brier_seen_mean', 'brier_seen_std', 
                                    'brier_unseen_mean', 'brier_unseen_std'])
    
    for benchmark in ROC_FILES.keys():
        # Load ROC-AUC data
        roc_path = os.path.join(DATA_DIR, ROC_FILES[benchmark])
        if not os.path.exists(roc_path):
            print(f"Warning: {roc_path} not found, skipping {benchmark}")
            continue
        
        roc_df = pd.read_csv(roc_path)
        
        # Get Brier scores for this benchmark
        brier_benchmark = brier_df[brier_df['benchmark'] == benchmark]
        
        for _, brier_row in brier_benchmark.iterrows():
            method = brier_row['method']
            
            # Find matching ROC-AUC
            roc_match = roc_df[roc_df['run_name'] == method]
            
            if not roc_match.empty:
                is_td0 = is_td0_method(method)
                
                # Add unseen data point
                data_points.append({
                    'method': method,
                    'benchmark': benchmark,
                    'split': 'val_unseen',
                    'brier_score': brier_row['brier_unseen_mean'],
                    'brier_std': brier_row['brier_unseen_std'],
                    'roc_auc': roc_match.iloc[0]['model_val_unseen_mean'],
                    'roc_auc_std': roc_match.iloc[0]['model_val_unseen_std'],
                    'color': get_method_color(method, is_td0),
                    'is_td0': is_td0
                })
                
                # Add seen data point
                data_points.append({
                    'method': method,
                    'benchmark': benchmark,
                    'split': 'val_seen',
                    'brier_score': brier_row['brier_seen_mean'],
                    'brier_std': brier_row['brier_seen_std'],
                    'roc_auc': roc_match.iloc[0]['model_val_seen_mean'],
                    'roc_auc_std': roc_match.iloc[0]['model_val_seen_std'],
                    'color': get_method_color(method, is_td0),
                    'is_td0': is_td0
                })
            else:
                print(f"Warning: No ROC-AUC match for {method} in {benchmark}")
    
    return pd.DataFrame(data_points)

def _build_legend_elements(include_fit_line=True):
    """Return standard legend elements for method/split."""
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    mlp_light = mcolors.to_hex(tuple(min(1.0, c + 0.4 * (1.0 - c)) for c in mcolors.to_rgb(COLOR_GROUPS['mlp'])))
    rnn_features_light = mcolors.to_hex(tuple(min(1.0, c + 0.4 * (1.0 - c)) for c in mcolors.to_rgb(COLOR_GROUPS['lstm'])))
    rnn_topk_light = mcolors.to_hex(tuple(min(1.0, c + 0.4 * (1.0 - c)) for c in mcolors.to_rgb(COLOR_GROUPS['qlearning'])))

    elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_GROUPS['mlp'],
               markersize=12, label='MLP feat (TDQC)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=mlp_light,
               markersize=12, label='MLP feat (BCE)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_GROUPS['lstm'],
               markersize=12, label='RNN feat (TDQC)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=rnn_features_light,
               markersize=12, label='RNN feat (BCE)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_GROUPS['qlearning'],
               markersize=12, label='RNN top-10 (TDQC)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=rnn_topk_light,
               markersize=12, label='RNN top-10 (BCE)', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], color='none', label=''),
        Line2D([0],[0], marker='o', color='gray', markersize=12, label='Unseen',
               markerfacecolor='lightgray', markeredgecolor='black', markeredgewidth=1.2),
        Line2D([0],[0], marker='^', color='gray', markersize=12, label='Seen',
               markerfacecolor='lightgray', markeredgecolor='black', markeredgewidth=1.2),
    ]
    if include_fit_line:
        elements += [
            Line2D([0],[0], color='none', label=''),
            Line2D([0],[0], color='red', linestyle='--', linewidth=3, label='Linear Fit'),
        ]
    return elements


def _scatter_points(ax, df_subset, alpha=0.8, size=150):
    """Plot seen/unseen points from df_subset onto ax."""
    for split in ['val_seen', 'val_unseen']:
        df_split = df_subset[df_subset['split'] == split]
        if df_split.empty:
            continue
        marker = 'o' if split == 'val_unseen' else '^'
        ax.scatter(df_split['brier_score'], df_split['roc_auc'],
                   c=df_split['color'], marker=marker, s=size, alpha=alpha,
                   edgecolors='black', linewidth=1.2, zorder=3)


def plot_brier_vs_roc(df, output_path, title=None):
    """Create scatter plot with fitted line and correlation."""
    fig, ax = plt.subplots(figsize=(12, 9))

    _scatter_points(ax, df)

    # Fit linear regression
    X = df['brier_score'].values.reshape(-1, 1)
    y = df['roc_auc'].values

    reg = LinearRegression()
    reg.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred  = reg.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred, 'r--', linewidth=3, alpha=0.8)

    rho, p_value = spearmanr(df['brier_score'], df['roc_auc'])

    ax.set_xlabel('Brier Score at Stop Time ($\\hat{T}$)', fontsize=20)
    ax.set_ylabel('ROC-AUC', fontsize=20)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=6)

    x_min, x_max = df['brier_score'].min(), df['brier_score'].max()
    x_padding = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    if title:
        ax.set_title(title, fontsize=22, pad=10)

    ax.legend(handles=_build_legend_elements(), fontsize=16, loc='upper right',
              frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    textstr = f'Spearman $\\rho$ = {rho:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.95, edgecolor='black', linewidth=2)
    ax.text(0.35, 0.98, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='left', bbox=props, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print(f"\nCorrelation Statistics ({title or 'all'}):")
    print(f"Spearman ρ = {rho:.4f}, p-value = {p_value:.4e}")
    print(f"Linear regression R² = {reg.score(X, y):.4f}")
    print(f"N={len(df)}  BCE={( ~df['is_td0']).sum()}  TD-0={df['is_td0'].sum()}")


def plot_brier_vs_roc_per_benchmark(df, all_df, output_dir):
    """For each benchmark produce a plot: grey background = all data, coloured = that benchmark."""
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in df['benchmark'].unique():
        df_bench = df[df['benchmark'] == benchmark]
        if df_bench.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 9))

        # --- background: all other data in light grey ---
        df_other = all_df[all_df['benchmark'] != benchmark]
        for split in ['val_seen', 'val_unseen']:
            df_split = df_other[df_other['split'] == split]
            if df_split.empty:
                continue
            marker = 'o' if split == 'val_unseen' else '^'
            ax.scatter(df_split['brier_score'], df_split['roc_auc'],
                       color='#cccccc', marker=marker, s=100, alpha=0.4,
                       edgecolors='#aaaaaa', linewidth=0.8, zorder=1)

        # --- foreground: this benchmark's points with full colour ---
        _scatter_points(ax, df_bench, alpha=0.9, size=200)

        # --- global regression line (using all data) ---
        X_all = all_df['brier_score'].values.reshape(-1, 1)
        y_all = all_df['roc_auc'].values
        reg_all = LinearRegression().fit(X_all, y_all)
        x_range = np.linspace(X_all.min(), X_all.max(), 100)
        ax.plot(x_range, reg_all.predict(x_range.reshape(-1, 1)),
                'r--', linewidth=2.5, alpha=0.6, zorder=2)

        # --- benchmark-specific regression (if enough points) ---
        if len(df_bench) >= 4:
            X_b = df_bench['brier_score'].values.reshape(-1, 1)
            y_b = df_bench['roc_auc'].values
            reg_b = LinearRegression().fit(X_b, y_b)
            x_b_range = np.linspace(X_b.min(), X_b.max(), 100)
            ax.plot(x_b_range, reg_b.predict(x_b_range.reshape(-1, 1)),
                    color='navy', linestyle='-', linewidth=2.5, alpha=0.8, zorder=4)

        rho, _ = spearmanr(df_bench['brier_score'], df_bench['roc_auc'])

        # clean benchmark name for file/title
        bname_safe = benchmark.replace('$', '').replace('\\', '').replace(' ', '_').replace('{', '').replace('}', '')

        ax.set_xlabel('Brier Score at Stop Time ($\\hat{T}$)', fontsize=20)
        ax.set_ylabel('ROC-AUC', fontsize=20)
        ax.set_title(benchmark, fontsize=22, pad=10)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=6)

        # x-axis covers full range so grey points provide context
        x_min, x_max = all_df['brier_score'].min(), all_df['brier_score'].max()
        x_padding = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - x_padding, x_max + x_padding)

        from matplotlib.lines import Line2D
        legend_els = _build_legend_elements(include_fit_line=False) + [
            Line2D([0],[0], color='none', label=''),
            Line2D([0],[0], color='#cccccc', marker='o', markersize=10, linestyle='None',
                   markeredgecolor='#aaaaaa', label='Other benchmarks'),
            Line2D([0],[0], color='r',    linestyle='--', linewidth=2.5, label='Global fit'),
        ]
        if len(df_bench) >= 4:
            legend_els.append(
                Line2D([0],[0], color='navy', linestyle='-', linewidth=2.5, label=f'{benchmark} fit')
            )

        ax.legend(handles=legend_els, fontsize=14, loc='upper right',
                  frameon=True, fancybox=True, shadow=True, framealpha=0.95)

        textstr = f'Spearman $\\rho$ = {rho:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.95, edgecolor='black', linewidth=2)
        ax.text(0.45, 0.98, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='left', bbox=props, weight='bold')

        out_path = os.path.join(output_dir, f'brier_vs_roc_{bname_safe}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    if df.empty:
        print("No data loaded. Check file paths.")
        return
    
    print(f"Loaded {len(df)} data points")
    print(f"Benchmarks: {df['benchmark'].unique()}")
    print(f"Splits: {df['split'].unique()}")
    
    # Overall plot (all benchmarks combined)
    output_path = os.path.join(DATA_DIR, 'brier_at_stop_vs_roc_scatter.png')
    plot_brier_vs_roc(df, output_path, title='All Benchmarks')

    # Per-benchmark plots (benchmark highlighted, rest greyed out)
    per_bench_dir = os.path.join(DATA_DIR, 'brier_vs_roc_per_benchmark')
    plot_brier_vs_roc_per_benchmark(df, df, per_bench_dir)
    
    # Save data to CSV
    csv_path = os.path.join(DATA_DIR, 'brier_at_stop_vs_roc_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved data to: {csv_path}")

if __name__ == '__main__':
    main()
