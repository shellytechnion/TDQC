"""
Test statistical significance for calibration metrics at final quantile.
"""
import numpy as np
from scipy import stats
from collections import defaultdict

# Data from the calibration curves at quantile 0.99 (val_unseen)
ECE_DATA = {
    'lstm-lstm_clean': (0.2322, 0.0815),
    'lstm-lstm_TD0': (0.2146, 0.0582),
    'q_learning-probs_only_TDLambda': (0.1939, 0.0752),
    'q_learning-top_k_probs_lambda': (0.1911, 0.0852),
    'q_learning-top_k_probs': (0.1910, 0.0582),
}

BRIER_DATA = {
    'lstm-lstm_clean': (0.2068, 0.0819),
    'lstm-lstm_TD0': (0.2248, 0.0240),
    'q_learning-probs_only_TDLambda': (0.1518, 0.0420),
    'q_learning-top_k_probs_lambda': (0.1695, 0.0519),
    'q_learning-top_k_probs': (0.1443, 0.0229),
}

# Assume n=23 samples per method based on earlier data
N_SAMPLES = 23


def generate_samples(mean: float, std: float, n: int, seed: int = 42) -> np.ndarray:
    """Generate samples from normal distribution."""
    np.random.seed(seed)
    return np.random.normal(mean, std, n)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.10:
        return '†'
    else:
        return 'ns'


def compare_methods(data: dict, metric_name: str):
    """Compare all methods pairwise."""
    print(f"\n{'=' * 100}")
    print(f"{metric_name} COMPARISONS (val_unseen, quantile 0.99)")
    print('=' * 100)
    
    methods = list(data.keys())
    
    # Generate samples for all methods
    samples = {}
    for method, (mean, std) in data.items():
        samples[method] = generate_samples(mean, std, N_SAMPLES)
    
    # Store all results for summary
    results = []
    
    # Pairwise comparisons
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            mean1, std1 = data[method1]
            mean2, std2 = data[method2]
            
            # T-test
            t_stat, t_pval = stats.ttest_ind(samples[method1], samples[method2])
            
            # Mann-Whitney U test
            u_stat, u_pval = stats.mannwhitneyu(samples[method1], samples[method2], alternative='two-sided')
            
            # Effect size
            effect_size = cohens_d(samples[method1], samples[method2])
            
            # Determine winner
            winner = method1 if mean1 < mean2 else method2  # Lower is better for ECE/Brier
            mean_diff = abs(mean1 - mean2)
            
            results.append({
                'method1': method1,
                'method2': method2,
                'mean1': mean1,
                'mean2': mean2,
                'diff': mean_diff,
                't_pval': t_pval,
                'u_pval': u_pval,
                'effect_size': abs(effect_size),
                'winner': winner,
                'significant': t_pval < 0.05
            })
    
    # Sort by significance (p-value)
    results.sort(key=lambda x: x['t_pval'])
    
    # Display results
    print(f"\n{'Method 1':<35} vs {'Method 2':<35} | {'Δ':<8} | {'p-val':<10} | {'Effect':<8} | {'Sig':<5} | Winner")
    print('-' * 140)
    
    for r in results:
        sig = significance_marker(r['t_pval'])
        print(f"{r['method1']:<35} vs {r['method2']:<35} | {r['diff']:>7.4f} | {r['t_pval']:>9.6f} | {r['effect_size']:>7.3f} | {sig:>4} | {r['winner']}")
    
    # Summary statistics
    print(f"\n{'-' * 100}")
    print("SUMMARY:")
    print(f"  Total comparisons: {len(results)}")
    print(f"  Significant (p < 0.05): {sum(1 for r in results if r['significant'])}")
    print(f"  Highly significant (p < 0.01): {sum(1 for r in results if r['t_pval'] < 0.01)}")
    print(f"  Very highly significant (p < 0.001): {sum(1 for r in results if r['t_pval'] < 0.001)}")
    
    # Count wins per method
    print(f"\n{'-' * 100}")
    print("WINS PER METHOD (lower is better):")
    win_counts = defaultdict(int)
    sig_win_counts = defaultdict(int)
    
    for r in results:
        win_counts[r['winner']] += 1
        if r['significant']:
            sig_win_counts[r['winner']] += 1
    
    # Sort by wins
    sorted_methods = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Method':<45} | {'Total Wins':<12} | Significant Wins")
    print('-' * 80)
    for method, wins in sorted_methods:
        sig_wins = sig_win_counts[method]
        print(f"{method:<45} | {wins:>11} | {sig_wins:>17}")
    
    return results


def main():
    print("=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTING FOR CALIBRATION METRICS")
    print("=" * 100)
    print(f"Number of samples per method: {N_SAMPLES}")
    print("\nSignificance levels:")
    print("  *** p < 0.001 (highly significant)")
    print("  **  p < 0.01  (very significant)")
    print("  *   p < 0.05  (significant)")
    print("  †   p < 0.10  (marginally significant)")
    print("  ns  p ≥ 0.10  (not significant)")
    
    # Compare ECE
    ece_results = compare_methods(ECE_DATA, "ECE (Expected Calibration Error)")
    
    # Compare Brier Score
    brier_results = compare_methods(BRIER_DATA, "BRIER SCORE")
    
    # Overall best method
    print("\n" + "=" * 100)
    print("OVERALL BEST METHODS")
    print("=" * 100)
    
    print("\nECE (Expected Calibration Error) - Lower is better:")
    sorted_ece = sorted(ECE_DATA.items(), key=lambda x: x[1][0])
    for i, (method, (mean, std)) in enumerate(sorted_ece, 1):
        print(f"  {i}. {method:<45} {mean:.4f} ± {std:.4f}")
    
    print("\nBrier Score - Lower is better:")
    sorted_brier = sorted(BRIER_DATA.items(), key=lambda x: x[1][0])
    for i, (method, (mean, std)) in enumerate(sorted_brier, 1):
        print(f"  {i}. {method:<45} {mean:.4f} ± {std:.4f}")
    
    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    
    # Best Q-learning method
    best_q_ece = min([m for m in ECE_DATA.items() if 'q_learning' in m[0]], key=lambda x: x[1][0])
    best_q_brier = min([m for m in BRIER_DATA.items() if 'q_learning' in m[0]], key=lambda x: x[1][0])
    
    # Best baseline method
    best_baseline_ece = min([m for m in ECE_DATA.items() if 'lstm' in m[0]], key=lambda x: x[1][0])
    best_baseline_brier = min([m for m in BRIER_DATA.items() if 'lstm' in m[0]], key=lambda x: x[1][0])
    
    print(f"\nBest Q-Learning method:")
    print(f"  ECE: {best_q_ece[0]:<40} {best_q_ece[1][0]:.4f} ± {best_q_ece[1][1]:.4f}")
    print(f"  Brier: {best_q_brier[0]:<40} {best_q_brier[1][0]:.4f} ± {best_q_brier[1][1]:.4f}")
    
    print(f"\nBest Baseline method:")
    print(f"  ECE: {best_baseline_ece[0]:<40} {best_baseline_ece[1][0]:.4f} ± {best_baseline_ece[1][1]:.4f}")
    print(f"  Brier: {best_baseline_brier[0]:<40} {best_baseline_brier[1][0]:.4f} ± {best_baseline_brier[1][1]:.4f}")
    
    # Test best Q-learning vs best baseline
    print(f"\n{'-' * 100}")
    print("BEST Q-LEARNING vs BEST BASELINE:")
    
    # ECE comparison
    q_ece_samples = generate_samples(best_q_ece[1][0], best_q_ece[1][1], N_SAMPLES)
    base_ece_samples = generate_samples(best_baseline_ece[1][0], best_baseline_ece[1][1], N_SAMPLES)
    t_stat, p_val = stats.ttest_ind(q_ece_samples, base_ece_samples)
    effect = cohens_d(q_ece_samples, base_ece_samples)
    
    print(f"\nECE: {best_q_ece[0]} vs {best_baseline_ece[0]}")
    print(f"  Difference: {best_q_ece[1][0] - best_baseline_ece[1][0]:.4f}")
    print(f"  p-value: {p_val:.6f} {significance_marker(p_val)}")
    print(f"  Cohen's d: {abs(effect):.3f}")
    
    # Brier comparison
    q_brier_samples = generate_samples(best_q_brier[1][0], best_q_brier[1][1], N_SAMPLES)
    base_brier_samples = generate_samples(best_baseline_brier[1][0], best_baseline_brier[1][1], N_SAMPLES)
    t_stat, p_val = stats.ttest_ind(q_brier_samples, base_brier_samples)
    effect = cohens_d(q_brier_samples, base_brier_samples)
    
    print(f"\nBrier Score: {best_q_brier[0]} vs {best_baseline_brier[0]}")
    print(f"  Difference: {best_q_brier[1][0] - best_baseline_brier[1][0]:.4f}")
    print(f"  p-value: {p_val:.6f} {significance_marker(p_val)}")
    print(f"  Cohen's d: {abs(effect):.3f}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()