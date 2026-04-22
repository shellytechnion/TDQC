#!/usr/bin/env python3
"""
Analyze conformal prediction results from videos_functional_cp directory.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def analyze_predictions(videos_dir):
    """
    Analyze prediction outcomes from directory names.
    Format: task{N}_ep{M}_succ{0|1}_predsucc{0|1}
    """
    # Count outcomes overall and per task
    overall = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    by_task = defaultdict(lambda: {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0})
    
    # Pattern to match directory names
    pattern = re.compile(r'task(\d+)_ep\d+_succ([01])_predsucc([01])')
    
    for entry in os.listdir(videos_dir):
        # Only process directories
        if not os.path.isdir(os.path.join(videos_dir, entry)):
            continue
            
        match = pattern.match(entry)
        if match:
            task_id = int(match.group(1))
            actual_success = int(match.group(2))
            predicted_success = int(match.group(3))
            
            if actual_success == 0 and predicted_success == 0:
                overall['TN'] += 1
                by_task[task_id]['TN'] += 1
            elif actual_success == 0 and predicted_success == 1:
                overall['FN'] += 1
                by_task[task_id]['FN'] += 1
            elif actual_success == 1 and predicted_success == 0:
                overall['FP'] += 1
                by_task[task_id]['FP'] += 1
            elif actual_success == 1 and predicted_success == 1:
                overall['TP'] += 1
                by_task[task_id]['TP'] += 1
    
    def print_task_stats(task_id, stats):
        """Print statistics for a single task."""
        tn, fp, fn, tp = stats['TN'], stats['FP'], stats['FN'], stats['TP']
        total = tn + fp + fn + tp
        
        if total == 0:
            return
        
        correct = tn + tp
        incorrect = fp + fn
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"TASK {task_id} ANALYSIS ({total} episodes)")
        print(f"{'='*70}")
        
        print("\nConfusion Matrix:")
        print("-" * 70)
        print(f"{'':25} | {'Predicted Failure (0)':20} | {'Predicted Success (1)':20}")
        print("-" * 70)
        print(f"{'Actual Failure (0)':25} | {f'TN: {tn}':20} | {f'FN: {fn}':20}")
        print(f"{'Actual Success (1)':25} | {f'FP: {fp}':20} | {f'TP: {tp}':20}")
        print("-" * 70)
        
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Incorrect: {incorrect}/{total} ({100-accuracy:.1f}%)")
        
        if incorrect > 0:
            fp_pct = (fp / incorrect * 100)
            fn_pct = (fn / incorrect * 100)
            print(f"\nError Breakdown:")
            print(f"  False Positives (predicted failure but was success): {fp}/{incorrect} ({fp_pct:.1f}%)")
            print(f"  False Negatives (predicted success but was failure): {fn}/{incorrect} ({fn_pct:.1f}%)")
        
        # Additional metrics
        actual_failures = tn + fn
        actual_successes = fp + tp
        
        if actual_failures > 0:
            tnr = tn / actual_failures * 100
            print(f"\nTrue Negative Rate (Specificity): {tnr:.1f}% ({tn}/{actual_failures})")
        
        if actual_successes > 0:
            tpr = tp / actual_successes * 100
            print(f"True Positive Rate (Sensitivity): {tpr:.1f}% ({tp}/{actual_successes})")
        
        predicted_failures = tn + fp
        predicted_successes = fn + tp
        
        if predicted_failures > 0:
            ppv_failure = tn / predicted_failures * 100
            print(f"Precision for Failure prediction: {ppv_failure:.1f}% ({tn}/{predicted_failures})")
        
        if predicted_successes > 0:
            ppv_success = tp / predicted_successes * 100
            print(f"Precision for Success prediction: {ppv_success:.1f}% ({tp}/{predicted_successes})")
    
    # Print per-task statistics for tasks 3, 4, 9
    for task_id in sorted([3, 4, 9]):
        if task_id in by_task:
            print_task_stats(task_id, by_task[task_id])
    
    # Print overall statistics
    tn, fp, fn, tp = overall['TN'], overall['FP'], overall['FN'], overall['TP']
    total = tn + fp + fn + tp
    
    # Calculate metrics
    correct_predictions = tn + tp
    incorrect_predictions = fp + fn
    
    print(f"\n{'='*70}")
    print("OVERALL ANALYSIS (ALL TASKS)")
    print(f"{'='*70}")
    print(f"\nTotal episodes: {total}")
    print()
    
    print("Confusion Matrix:")
    print("-" * 70)
    print(f"{'':25} | {'Predicted Failure (0)':20} | {'Predicted Success (1)':20}")
    print("-" * 70)
    print(f"{'Actual Failure (0)':25} | {f'TN: {tn}':20} | {f'FN: {fn}':20}")
    print(f"{'Actual Success (1)':25} | {f'FP: {fp}':20} | {f'TP: {tp}':20}")
    print("-" * 70)
    print()
    
    print("Overall Accuracy:")
    print("-" * 70)
    accuracy = (correct_predictions / total * 100) if total > 0 else 0
    print(f"Correct predictions:   {correct_predictions}/{total} ({accuracy:.1f}%)")
    print(f"Incorrect predictions: {incorrect_predictions}/{total} ({100-accuracy:.1f}%)")
    print()
    
    print("Error Breakdown:")
    print("-" * 70)
    if incorrect_predictions > 0:
        fp_pct = (fp / incorrect_predictions * 100)
        fn_pct = (fn / incorrect_predictions * 100)
        print(f"False Positives (predicted failure but was success):  {fp}/{incorrect_predictions} ({fp_pct:.1f}%)")
        print(f"False Negatives (predicted success but was failure):  {fn}/{incorrect_predictions} ({fn_pct:.1f}%)")
    else:
        print("No incorrect predictions!")
    print()
    
    # Additional metrics
    print("Additional Metrics:")
    print("-" * 70)
    
    actual_failures = tn + fn
    actual_successes = fp + tp
    
    if actual_failures > 0:
        tnr = tn / actual_failures * 100
        print(f"True Negative Rate (TNR/Specificity):  {tnr:.1f}% ({tn}/{actual_failures} failures correctly predicted)")
    
    if actual_successes > 0:
        tpr = tp / actual_successes * 100
        print(f"True Positive Rate (TPR/Sensitivity):  {tpr:.1f}% ({tp}/{actual_successes} successes correctly predicted)")
    
    predicted_failures = tn + fp
    predicted_successes = fn + tp
    
    if predicted_failures > 0:
        ppv_failure = tn / predicted_failures * 100
        print(f"Precision for Failure prediction:      {ppv_failure:.1f}% ({tn}/{predicted_failures})")
    
    if predicted_successes > 0:
        ppv_success = tp / predicted_successes * 100
        print(f"Precision for Success prediction:      {ppv_success:.1f}% ({tp}/{predicted_successes})")
    
    print("=" * 70)
    
    return {
        'total': total,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'accuracy': accuracy,
        'by_task': dict(by_task)
    }


if __name__ == "__main__":
    videos_dir = str(Path(__file__).resolve().parents[1] / "videos_functional_cp")
    
    if not os.path.exists(videos_dir):
        print(f"Error: Directory not found: {videos_dir}")
        exit(1)
    
    analyze_predictions(videos_dir)
