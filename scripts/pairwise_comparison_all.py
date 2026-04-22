"""
Pairwise statistical comparison of all methods with significance testing.
"""
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# Data from user (n=23 for all methods)
data = {
    'indep-mlp_clean': {
        'seen': np.array([0.7208964515459638, 0.7801592161665645, 0.6582687338501292, 0.6939487179487179, 0.8298402158124092, 0.7060416666666667, 0.7190758536086691, 0.7918709150326797, 0.7754415475189234, 0.7254901960784313, 0.7663054590063382, 0.7468982630272953, 0.6825657894736841, 0.7171160130718953, 0.7009925558312655, 0.7474106954132319, 0.717516447368421, 0.7491106926135174, 0.75, 0.7275326797385621, 0.7414922849291905]),
        'unseen': np.array([0.7575259989053093, 0.7437499999999999, 0.722495894909688, 0.5782271241830066, 0.6128185907046477, 0.6615384615384615, 0.7270275130372235, 0.6463643178410794, 0.6764814814814815, 0.6439491619082705, 0.6988080412737947, 0.7456253453674709, 0.6410806942207908, 0.7352469245854876, 0.7270923520923521, 0.7634159386699947, 0.702887537993921, 0.7188833570412517, 0.7537339971550497, 0.6695195487259288, 0.6747098913243691])
    },
    'lstm-lstm_clean': {
        'seen': np.array([0.7447603237186138, 0.7358644621351296, 0.773811994189666, 0.6255383290267011, 0.6894358974358974, 0.7402083333333334, 0.6381108157840933, 0.7988153594771242, 0.7693439865433137, 0.7230392156862745, 0.7720302596606011, 0.7332506203473946, 0.6889391447368421, 0.7489787581699346, 0.7187758478081059, 0.6878038469668146, 0.6780427631578947, 0.7281858129315757, 0.7485608552631577, 0.6511437908496731, 0.7886281969985204]),
        'unseen': np.array([0.7735814632366356, 0.7587499999999999, 0.5757121439280359, 0.8337894544791097, 0.5859885620915033, 0.6428959276018099, 0.5905412695558353, 0.5787106446776611, 0.6931481481481482, 0.7261005710075522, 0.7692581391211528, 0.6463437097071283, 0.7371622830560028, 0.6840791584952753, 0.6691919191919192, 0.7505794259226244, 0.6371580547112463, 0.7112375533428166, 0.745199146514936, 0.6829410620501848, 0.7056548167249953])
    },
    'lstm-lstm_TD0': {
        'seen': np.array([0.7121809504046482, 0.7764849969381505, 0.8113716538700976, 0.6457795004306632, 0.6867692307692308, 0.6868749999999999, 0.7192803107748926, 0.7914624183006537, 0.7302354920100926, 0.7573529411764706, 0.7732570026579432, 0.750620347394541, 0.697985197368421, 0.7399918300653595, 0.7688172043010754, 0.720566476432044, 0.7588404605263157, 0.7106089139987445, 0.7477384868421053, 0.7375408496732027, 0.7150708095540056]),
        'unseen': np.array([0.7088122605363985, 0.7853571428571429, 0.6664167916041979, 0.7489509213647145, 0.6288807189542485, 0.6776470588235294, 0.7045495414493796, 0.6349325337331333, 0.7064814814814815, 0.638975870326027, 0.6239103362391033, 0.7074967765702707, 0.6716765074252997, 0.6441433410590123, 0.7251082251082251, 0.7536102692101979, 0.6476063829787234, 0.6504267425320056, 0.7498221906116643, 0.6029955261622253, 0.605452201142015])
    },
    'q_learning-probs_only_TDLambda': {
        'seen': np.array([0.7121809504046482, 0.6901408450704225, 0.7395725254202116, 0.7263135228251507, 0.7595897435897436, 0.7627083333333334, 0.7744837456552852, 0.7732843137254902, 0.7323380992430614, 0.6948529411764706, 0.7340012267429975, 0.717741935483871, 0.618421052631579, 0.7424428104575164, 0.7140198511166254, 0.6605368843796237, 0.768092105263158, 0.7432517263025739, 0.6303453947368421, 0.6911764705882353, 0.7176072711900232]),
        'unseen': np.array([0.7447546068235723, 0.7867857142857143, 0.758808095952024, 0.7431125706987778, 0.6631944444444444, 0.7087782805429864, 0.7741413414853444, 0.6847826086956522, 0.6751851851851852, 0.7325474304660157, 0.7345668030599537, 0.6524221771965371, 0.7733047056718554, 0.7618113745765734, 0.601911976911977, 0.7523622749153147, 0.7287234042553191, 0.6847439544807966, 0.738620199146515, 0.7582182454775336, 0.7251795910849144])
    },
    'q_learning-probs_only_TD0': {
        'seen': np.array([0.733139655530193, 0.6731986119616248, 0.7920730442000415, 0.7269595176571921, 0.7302564102564103, 0.7150967199327165, 0.741421568627451, 0.8020833333333334, 0.7869556327949294, 0.7487593052109182, 0.7016969944796565, 0.7029194078947368, 0.7371323529411765, 0.7865604575163399, 0.7297353184449958, 0.6383428450644684, 0.748560855263158, 0.7078886796400922, 0.7060032894736842, 0.6877042483660131, 0.7848235045444938]),
        'unseen': np.array([0.7713920817369093, 0.7894642857142857, 0.6971514242878559, 0.7715745301952199, 0.6776960784313726, 0.6994444444444444, 0.7605452201142017, 0.7261538461538461, 0.7523572318092866, 0.6253453674709891, 0.7762992267577773, 0.7702630166398281, 0.6371814092953524, 0.7477268675343198, 0.6924603174603176, 0.7470137279372437, 0.7281534954407295, 0.6381578947368421, 0.7519559032716927, 0.7702781560007781, 0.769570823356051])
    },
    'q_learning-top_k_probs_lambda': {
        'seen': np.array([0.7444102564102564, 0.720351092059604, 0.7779622328283876, 0.6907838070628769, 0.7714871794871795, 0.7208333333333333, 0.720302596606011, 0.7816584967320261, 0.7417998317914214, 0.6650326797385622, 0.7192803107748927, 0.7156741108354011, 0.6414473684210525, 0.7908496732026143, 0.7231182795698925, 0.6785034876347494, 0.7232730263157895, 0.7055869428750785, 0.6245888157894737, 0.7455065359477124, 0.7288099767491016]),
        'unseen': np.array([0.7210363072432038, 0.7905357142857142, 0.7003373313343328, 0.745666849115125, 0.7261029411764706, 0.6691402714932125, 0.8023736737996763, 0.6519865067466266, 0.7116666666666668, 0.7395468778780623, 0.6975627112613414, 0.6715785595874011, 0.7462873501520844, 0.7311463719022998, 0.7085137085137085, 0.7719736138349083, 0.6728723404255319, 0.6539829302987198, 0.7384423897581792, 0.7385722622057965, 0.8064100202615583])
    },
    'q_learning-top_k_probs': {
        'seen': np.array([0.7187692307692308, 0.6993263931414574, 0.7885453413571281, 0.7077950043066322, 0.7702564102564102, 0.7741666666666667, 0.7566959721938253, 0.7704248366013072, 0.7308662741799832, 0.7395833333333334, 0.7832754038029034, 0.7768817204301076, 0.681126644736842, 0.7579656862745098, 0.7146401985111662, 0.6647643204396533, 0.7674753289473684, 0.7288135593220338, 0.7271792763157895, 0.7083333333333334, 0.7391671950961741]),
        'unseen': np.array([0.729428936325488, 0.7864285714285714, 0.7331334332833583, 0.7765006385696042, 0.7228349673202615, 0.709683257918552, 0.7897860097104837, 0.6431784107946027, 0.7098148148148148, 0.7798857984895929, 0.7607187333214731, 0.668815619819488, 0.7795670066201467, 0.7618113745765734, 0.7101370851370851, 0.7108218933856302, 0.7009878419452888, 0.6331792318634424, 0.7585348506401137, 0.755106010503793, 0.8005157487566771])
    }
}


def perform_pairwise_comparisons(data: dict, split: str, alpha: float = 0.05):
    """
    Perform all pairwise comparisons with multiple significance levels.
    """
    print(f"\n{'=' * 120}")
    print(f"PAIRWISE COMPARISONS: {split.upper()} SPLIT")
    print("=" * 120)
    
    method_names = list(data.keys())
    results = []
    
    # Perform all pairwise comparisons
    for method1, method2 in combinations(method_names, 2):
        values1 = data[method1][split]
        values2 = data[method2][split]
        
        n1, n2 = len(values1), len(values2)
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
        sem1, sem2 = std1 / np.sqrt(n1), std2 / np.sqrt(n2)
        
        # Two-sided t-test
        t_stat, p_two_sided = stats.ttest_ind(values1, values2)
        
        # One-sided t-tests
        t_stat_greater, p_1_greater_2 = stats.ttest_ind(values1, values2, alternative='greater')
        t_stat_less, p_2_greater_1 = stats.ttest_ind(values1, values2, alternative='less')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # 95% confidence intervals
        ci1 = stats.t.interval(0.95, n1-1, loc=mean1, scale=sem1)
        ci2 = stats.t.interval(0.95, n2-1, loc=mean2, scale=sem2)
        
        # Determine significance at different levels
        sig_001 = p_two_sided < 0.001
        sig_01 = p_two_sided < 0.01
        sig_05 = p_two_sided < 0.05
        sig_10 = p_two_sided < 0.10
        
        # Significance marker
        if sig_001:
            sig_marker = '***'
        elif sig_01:
            sig_marker = '**'
        elif sig_05:
            sig_marker = '*'
        elif sig_10:
            sig_marker = '†'
        else:
            sig_marker = 'ns'
        
        # Better method
        if mean1 > mean2:
            better_method = method1
            better_mean = mean1
            worse_method = method2
            worse_mean = mean2
            p_better = p_1_greater_2
        else:
            better_method = method2
            better_mean = mean2
            worse_method = method1
            worse_mean = mean1
            p_better = p_2_greater_1
        
        results.append({
            'method_1': method1,
            'method_2': method2,
            'mean_1': mean1,
            'std_1': std1,
            'ci_1_lower': ci1[0],
            'ci_1_upper': ci1[1],
            'n_1': n1,
            'mean_2': mean2,
            'std_2': std2,
            'ci_2_lower': ci2[0],
            'ci_2_upper': ci2[1],
            'n_2': n2,
            'diff': mean1 - mean2,
            'abs_diff': abs(mean1 - mean2),
            'p_two_sided': p_two_sided,
            'p_1>2': p_1_greater_2,
            'p_2>1': p_2_greater_1,
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'sig_marker': sig_marker,
            'sig_0.001': sig_001,
            'sig_0.01': sig_01,
            'sig_0.05': sig_05,
            'sig_0.10': sig_10,
            'better_method': better_method,
            'p_better': p_better
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('abs_diff', ascending=False)
    
    # Display full comparison table
    print("\nFULL PAIRWISE COMPARISON TABLE:")
    print("-" * 120)
    display_cols = ['method_1', 'method_2', 'mean_1', 'mean_2', 'diff', 'p_two_sided', 'sig_marker', 'cohens_d']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Significance summary
    print("\n\nSIGNIFICANCE LEVELS:")
    print("-" * 120)
    print(f"  *** p < 0.001 (highly significant)")
    print(f"  **  p < 0.01  (very significant)")
    print(f"  *   p < 0.05  (significant)")
    print(f"  †   p < 0.10  (marginally significant)")
    print(f"  ns  p ≥ 0.10  (not significant)")
    
    # Count significance levels
    print(f"\nSignificance counts:")
    print(f"  p < 0.001: {df['sig_0.001'].sum()} comparisons")
    print(f"  p < 0.01:  {df['sig_0.01'].sum()} comparisons")
    print(f"  p < 0.05:  {df['sig_0.05'].sum()} comparisons")
    print(f"  p < 0.10:  {df['sig_0.10'].sum()} comparisons")
    print(f"  Not sig:   {(~df['sig_0.10']).sum()} comparisons")
    
    # Show significant comparisons
    print("\n\nSIGNIFICANT COMPARISONS (p < 0.05):")
    print("-" * 120)
    sig_df = df[df['sig_0.05']]
    if len(sig_df) > 0:
        for _, row in sig_df.iterrows():
            better = row['better_method']
            worse = row['worse_method'] if row['method_1'] == better else row['method_1']
            print(f"{row['sig_marker']:3s} {better:35s} > {worse:35s}  "
                  f"Δ={row['abs_diff']:.4f}  p={row['p_two_sided']:.4f}  d={row['cohens_d']:.3f}")
    else:
        print("No significant differences at p < 0.05")
    
    # Show marginally significant comparisons
    print("\n\nMARGINALLY SIGNIFICANT COMPARISONS (0.05 ≤ p < 0.10):")
    print("-" * 120)
    marginal_df = df[df['sig_0.10'] & ~df['sig_0.05']]
    if len(marginal_df) > 0:
        for _, row in marginal_df.iterrows():
            better = row['better_method']
            worse = row['worse_method'] if row['method_1'] == better else row['method_1']
            print(f"{row['sig_marker']:3s} {better:35s} > {worse:35s}  "
                  f"Δ={row['abs_diff']:.4f}  p={row['p_two_sided']:.4f}  d={row['cohens_d']:.3f}")
    else:
        print("No marginally significant differences")
    
    # Method ranking with win/loss record
    print("\n\nMETHOD PERFORMANCE SUMMARY:")
    print("-" * 120)
    method_summary = []
    
    for method in method_names:
        values = data[method][split]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        wins = 0
        sig_wins = 0
        losses = 0
        sig_losses = 0
        
        for _, row in df.iterrows():
            if row['method_1'] == method:
                if row['mean_1'] > row['mean_2']:
                    wins += 1
                    if row['sig_0.05']:
                        sig_wins += 1
                else:
                    losses += 1
                    if row['sig_0.05']:
                        sig_losses += 1
            elif row['method_2'] == method:
                if row['mean_2'] > row['mean_1']:
                    wins += 1
                    if row['sig_0.05']:
                        sig_wins += 1
                else:
                    losses += 1
                    if row['sig_0.05']:
                        sig_losses += 1
        
        method_summary.append({
            'method': method,
            'mean': mean_val,
            'std': std_val,
            'n': len(values),
            'wins': wins,
            'sig_wins': sig_wins,
            'losses': losses,
            'sig_losses': sig_losses,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0
        })
    
    summary_df = pd.DataFrame(method_summary)
    summary_df = summary_df.sort_values('mean', ascending=False)
    summary_df['rank'] = range(1, len(summary_df) + 1)
    
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Save to CSV
    output_file = f'./scripts/pairwise_comparison_{split}.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n\nFull results saved to: {output_file}")
    
    return df, summary_df


def main():
    print("=" * 120)
    print("COMPREHENSIVE PAIRWISE STATISTICAL COMPARISON")
    print("=" * 120)
    
    # Analyze both splits
    seen_df, seen_summary = perform_pairwise_comparisons(data, 'seen')
    unseen_df, unseen_summary = perform_pairwise_comparisons(data, 'unseen')
    
    # Overall recommendation
    print("\n\n" + "=" * 120)
    print("OVERALL RECOMMENDATION")
    print("=" * 120)
    
    # Calculate overall scores
    overall_scores = []
    for method in data.keys():
        seen_rank = seen_summary[seen_summary['method'] == method]['rank'].values[0]
        unseen_rank = unseen_summary[unseen_summary['method'] == method]['rank'].values[0]
        seen_mean = seen_summary[seen_summary['method'] == method]['mean'].values[0]
        unseen_mean = unseen_summary[unseen_summary['method'] == method]['mean'].values[0]
        seen_sig_wins = seen_summary[seen_summary['method'] == method]['sig_wins'].values[0]
        unseen_sig_wins = unseen_summary[unseen_summary['method'] == method]['sig_wins'].values[0]
        
        avg_rank = (seen_rank + unseen_rank) / 2
        avg_mean = (seen_mean + unseen_mean) / 2
        total_sig_wins = seen_sig_wins + unseen_sig_wins
        
        overall_scores.append({
            'method': method,
            'avg_rank': avg_rank,
            'avg_mean': avg_mean,
            'seen_rank': seen_rank,
            'unseen_rank': unseen_rank,
            'seen_mean': seen_mean,
            'unseen_mean': unseen_mean,
            'total_sig_wins': total_sig_wins
        })
    
    overall_df = pd.DataFrame(overall_scores)
    overall_df = overall_df.sort_values('avg_rank')
    
    print("\nOVERALL RANKINGS:")
    print("-" * 120)
    print(overall_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    best_method = overall_df.iloc[0]['method']
    print(f"\n\n🏆 RECOMMENDED BEST METHOD: {best_method}")
    print(f"   Average rank: {overall_df.iloc[0]['avg_rank']:.1f}")
    print(f"   Average ROC AUC: {overall_df.iloc[0]['avg_mean']:.4f}")
    print(f"   Total significant wins: {int(overall_df.iloc[0]['total_sig_wins'])}")
    print(f"   Seen:   {overall_df.iloc[0]['seen_mean']:.4f} (rank {int(overall_df.iloc[0]['seen_rank'])})")
    print(f"   Unseen: {overall_df.iloc[0]['unseen_mean']:.4f} (rank {int(overall_df.iloc[0]['unseen_rank'])})")


if __name__ == "__main__":
    main()
