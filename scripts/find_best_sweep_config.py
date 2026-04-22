"""
Find the best configuration from W&B sweep runs by averaging over seeds.

This script:
1. Fetches all runs matching a specific sweep name from W&B
2. Groups runs by configuration (excluding seed)
3. Averages metrics across seeds for each configuration
4. Identifies the best configuration based on a target metric
"""

"""
# Basic usage - find best config for a sweep
python scripts/find_best_sweep_config.py \
    --sweep-name "pizero-default-lstm-lstm_sweep" \
    --primary-metric "falert_early_roc_auc/model_val_seen"

# Process multiple sweeps in one run
python scripts/find_best_sweep_config.py \
    --sweep-names "pizero-default-lstm-lstm_sweep" "pizero-default-lstm-TD0_sweep" \
    --primary-metric "falert_early_roc_auc/model_val_seen" \
    --output "./sweep_results.csv"

# Save results to file
python scripts/find_best_sweep_config.py \
    --sweep-name "pizero-default-lstm-lstm_sweep" \
    --output "./sweep_results.csv"

# Minimize a metric instead of maximize (e.g., for loss or Brier score)
python scripts/find_best_sweep_config.py \
    --sweep-name "pizero-default-lstm-lstm_TD0_sweep" \
    --primary-metric "calibration/model_brier_at_stop_val_seen" \
    --no-maximize \
    --output "./sweep_results.csv"

# Customize metrics to extract
python scripts/find_best_sweep_config.py \
    --sweep-name "pizero-default-lstm-lstm_sweep" \
    --metrics "falert_early_roc_auc/model_val_seen" \
             "falert_early_roc_auc/model_val_seen" \
             "calibration/model_ece_at_stop_val_seen"

# Require at least 3 seeds per configuration
python scripts/find_best_sweep_config.py \
    --sweep-name "pizero-default-lstm-lstm_sweep" \
    --min-seeds 3
    """
import argparse
import numpy as np
import pandas as pd
import wandb
from collections import defaultdict
from typing import Dict, List, Tuple
import json

# Initialize the API with longer timeout
api = wandb.Api(timeout=60)
WANDB_USERNAME = api.viewer.username


def get_runs(project_name: str, sweep_name: str) -> list:
    """Fetch runs from W&B API matching the sweep name."""
    print(f"Fetching runs from project: {project_name}")
    print(f"Filtering for sweep name: {sweep_name}")
    all_runs = list(api.runs(project_name))
    
    # Filter runs by sweep name
    filtered_runs = [r for r in all_runs if r.name.startswith(sweep_name)]
    print(f"Found {len(filtered_runs)} runs matching '{sweep_name}'")
    return filtered_runs


def get_config_signature(config: dict, exclude_keys: set = None) -> str:
    """
    Create a unique signature for a configuration, excluding specified keys.
    
    Args:
        config: Configuration dictionary
        exclude_keys: Set of keys to exclude (e.g., {'seed', 'exp_name'})
    
    Returns:
        String signature of the configuration
    """
    if exclude_keys is None:
        exclude_keys = {'seed', 'exp_name', 'exp_suffix', 'wandb_group_name', 'run_id', 'wandb_run_id', 'run_name', 'timestamp'}
    
    def filter_dict_recursive(d):
        """Recursively filter out excluded keys from nested dictionaries."""
        if not isinstance(d, dict):
            return d
        
        filtered = {}
        for key, value in d.items():
            if key not in exclude_keys:
                if isinstance(value, dict):
                    # Recursively filter nested dicts
                    filtered[key] = filter_dict_recursive(value)
                elif isinstance(value, list):
                    # Handle lists that might contain dicts
                    filtered[key] = [filter_dict_recursive(item) if isinstance(item, dict) else item for item in value]
                else:
                    filtered[key] = value
        return filtered
    
    # Create a filtered config dict
    filtered_config = filter_dict_recursive(config)
    
    # Convert to sorted JSON string for consistent hashing
    return json.dumps(filtered_config, sort_keys=True)


def group_runs_by_config(runs: list, exclude_keys: set = None, debug: bool = False) -> Dict[str, List[wandb.apis.public.Run]]:
    """
    Group runs by their configuration (excluding seed and other specified keys).
    
    Returns:
        Dictionary mapping config_signature -> list of runs
    """
    if exclude_keys is None:
        exclude_keys = set()

    config_groups = defaultdict(list)
    
    for run in runs:
        try:
            config = run.config
            exclude_keys.add("td_horizon")
            config_sig = get_config_signature(config, exclude_keys)
            config_groups[config_sig].append(run)
            
            if debug:
                print(f"\nRun: {run.name}")
                print(f"  Seed: {config.get('train', {}).get('seed', 'N/A')}")
                print(f"  Config signature (first 200 chars): {config_sig[:200]}...")
        except Exception as e:
            print(f"Warning: Could not process run {run.name}: {e}")
            continue
    
    print(f"\nFound {len(config_groups)} unique configurations")
    
    if debug and len(config_groups) > 1:
        print("\n" + "=" * 80)
        print("DEBUG: Comparing first two config signatures to find differences")
        print("=" * 80)
        sigs = list(config_groups.keys())[:2]
        if len(sigs) == 2:
            import difflib
            diff = difflib.unified_diff(
                sigs[0].split(','),
                sigs[1].split(','),
                lineterm='',
                n=0
            )
            print('\n'.join(diff))
        print("=" * 80)
    
    for config_sig, runs_list in config_groups.items():
        seeds = [r.config.get('train', {}).get('seed', 'unknown') for r in runs_list]
        print(f"  Config with seeds {seeds}: {len(runs_list)} runs")
        if debug:
            print(f"    Run names: {[r.name for r in runs_list]}")
    
    return dict(config_groups)


def extract_metrics_for_config_group(
    runs: List[wandb.apis.public.Run],
    metrics: List[str]
) -> Dict[str, Tuple[float, float, int]]:
    """
    Extract and average metrics across runs in a configuration group.
    
    Args:
        runs: List of runs with the same configuration
        metrics: List of metric names to extract
    
    Returns:
        Dictionary mapping metric_name -> (mean, std, n_runs)
    """
    results = defaultdict(list)
    
    for run in runs:
        summary = run.summary._json_dict
        for metric in metrics:
            if metric in summary:
                try:
                    value = float(summary[metric])
                    results[metric].append(value)
                except (ValueError, TypeError):
                    continue
    
    # Calculate statistics
    stats = {}
    for metric, values in results.items():
        if values:
            stats[metric] = (np.mean(values), np.std(values), len(values))
        else:
            stats[metric] = (float('nan'), float('nan'), 0)
    
    return stats


def find_best_configuration(
    config_groups: Dict[str, List[wandb.apis.public.Run]],
    metrics: List[str],
    primary_metric: str,
    maximize: bool = True,
    min_seeds: int = 2
) -> Tuple[str, Dict, Dict]:
    """
    Find the best configuration based on averaged metrics.
    
    Args:
        config_groups: Dictionary mapping config_signature -> list of runs
        metrics: List of metrics to extract
        primary_metric: Metric to use for ranking configurations
        maximize: If True, higher is better; if False, lower is better
        min_seeds: Minimum number of seeds required for a configuration to be considered
    
    Returns:
        Tuple of (best_config_signature, best_config_dict, all_results_dict)
    """
    all_results = []
    
    for config_sig, runs_list in config_groups.items():
        # Skip configurations with insufficient seeds
        if len(runs_list) < min_seeds:
            print(f"Skipping config with only {len(runs_list)} seeds (min required: {min_seeds})")
            continue
        
        # Extract metrics
        metric_stats = extract_metrics_for_config_group(runs_list, metrics)
        
        # Get the configuration (use first run as representative)
        config = runs_list[0].config
        seeds = [r.config.get('train', {}).get('seed', 'unknown') for r in runs_list]
        
        # Create result entry
        result = {
            'config_signature': config_sig,
            'config': config,
            'n_seeds': len(runs_list),
            'seeds': seeds,
            'run_names': [r.name for r in runs_list]
        }
        
        # Add metric statistics
        for metric in metrics:
            if metric in metric_stats:
                mean, std, n = metric_stats[metric]
                metric_short = metric.split('/')[-1]
                result[f'{metric_short}_mean'] = mean
                result[f'{metric_short}_std'] = std
                result[f'{metric_short}_n'] = n
            else:
                metric_short = metric.split('/')[-1]
                result[f'{metric_short}_mean'] = float('nan')
                result[f'{metric_short}_std'] = float('nan')
                result[f'{metric_short}_n'] = 0
        
        all_results.append(result)
    
    if not all_results:
        raise ValueError("No valid configurations found!")
    
    # Find best configuration based on primary metric
    primary_metric_short = primary_metric.split('/')[-1]
    primary_metric_key = f'{primary_metric_short}_mean'
    
    # Filter out configs with NaN values for primary metric
    valid_results = [r for r in all_results if not np.isnan(r[primary_metric_key])]
    
    if not valid_results:
        raise ValueError(f"No configurations have valid values for {primary_metric}")
    
    # Sort by primary metric
    valid_results.sort(key=lambda x: x[primary_metric_key], reverse=maximize)
    
    best_result = valid_results[0]
    
    return best_result['config_signature'], best_result['config'], all_results


def print_results_table(all_results: List[Dict], metrics: List[str], primary_metric: str):
    """Print a formatted table of all configuration results."""
    if not all_results:
        print("No results to display")
        return
    
    # Create DataFrame for nice formatting
    rows = []
    for result in all_results:
        row = {
            'n_seeds': result['n_seeds'],
            'seeds': str(result['seeds']),
        }
        
        # Add configuration parameters (extract key hyperparameters)
        config = result['config']
        
        # Add dataset parameters (HIGHLIGHTED)
        if 'dataset' in config:
            dataset_config = config['dataset']
            if 'horizon_idx_rel' in dataset_config:
                row['horizon_idx_rel'] = dataset_config['horizon_idx_rel']
            if 'diff_idx_rel' in dataset_config:
                row['diff_idx_rel'] = dataset_config['diff_idx_rel']
            if 'token_idx_rel' in dataset_config:
                row['token_idx_rel'] = dataset_config['token_idx_rel']
        
        # Add model parameters (HIGHLIGHTED)
        if 'model' in config:
            model_config = config['model']
            row['lr'] = model_config.get('lr', 'N/A')
            row['lambda_reg'] = model_config.get('lambda_reg', 'N/A')
            row['lr_gamma'] = model_config.get('lr_gamma', 'N/A')
            
            # Other useful params
            row['batch_size'] = model_config.get('batch_size', 'N/A')
            row['n_epochs'] = model_config.get('n_epochs', 'N/A')
            row['loss'] = model_config.get('loss', 'N/A')
            
            # Add model-specific params if they exist
            if 'td_horizon' in model_config:
                row['td_horizon'] = model_config['td_horizon']
            if 'lr_step_size' in model_config:
                row['lr_step_size'] = model_config['lr_step_size']
        
        # Add metrics
        for metric in metrics:
            metric_short = metric.split('/')[-1]
            mean_val = result.get(f'{metric_short}_mean', float('nan'))
            std_val = result.get(f'{metric_short}_std', float('nan'))
            row[metric_short] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by primary metric mean extracted from formatted "mean ± std" strings.
    primary_metric_short = primary_metric.split('/')[-1]
    if primary_metric_short in [col.replace(' ± ', '_').split('_')[0] for col in df.columns]:
        # Extract mean value for sorting
        sort_values = []
        for _, row in df.iterrows():
            metric_str = row.get(primary_metric_short, 'nan ± nan')
            mean_str = str(metric_str).split(' ± ')[0]
            try:
                sort_values.append(float(mean_str))
            except ValueError:
                sort_values.append(float('nan'))
        df['_sort_key'] = sort_values
        df = df.sort_values('_sort_key', ascending=False).drop(columns=['_sort_key'])
    
    print("\n" + "=" * 120)
    print("ALL CONFIGURATIONS (sorted by primary metric)")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)


def main(args: argparse.Namespace):
    project_name = f"{WANDB_USERNAME}/{args.project}"
    
    # Support both single sweep-name and multiple sweep-names
    sweep_names = args.sweep_names if hasattr(args, 'sweep_names') and args.sweep_names else [args.sweep_name]
    
    # Process each sweep
    all_sweep_results = {}
    
    for sweep_name in sweep_names:
        print("\n" + "=" * 120)
        print(f"PROCESSING SWEEP: {sweep_name}")
        print("=" * 120)
        
        # Get all runs matching the sweep name
        runs = get_runs(project_name, sweep_name)
        
        if not runs:
            print(f"No runs found matching '{sweep_name}'")
            continue
        
        # Group runs by configuration
        config_groups = group_runs_by_config(runs, exclude_keys=set(args.exclude_keys), debug=args.debug)
        
        # Find best configuration
        print(f"\nSearching for best configuration based on metric: {args.primary_metric}")
        print(f"Optimization direction: {'maximize' if args.maximize else 'minimize'}")
        print(f"Minimum seeds required: {args.min_seeds}")
        
        try:
            best_config_sig, best_config, all_results = find_best_configuration(
                config_groups,
                args.metrics,
                args.primary_metric,
                maximize=args.maximize,
                min_seeds=args.min_seeds
            )
            
            # Store results for this sweep
            all_sweep_results[sweep_name] = {
                'best_config_sig': best_config_sig,
                'best_config': best_config,
                'all_results': all_results
            }
            
            # Print results table
            print_results_table(all_results, args.metrics, args.primary_metric)
            
            # Print best configuration
            print("\n" + "=" * 120)
            print(f"BEST CONFIGURATION FOR: {sweep_name}")
            print("=" * 120)
            best_result = next(r for r in all_results if r['config_signature'] == best_config_sig)
            print(f"Number of seeds: {best_result['n_seeds']}")
            print(f"Seeds: {best_result['seeds']}")
            print(f"Run names: {best_result['run_names']}")
            
            # Print highlighted parameters
            print("\n" + "=" * 60)
            print("HIGHLIGHTED PARAMETERS")
            print("=" * 60)
            if 'dataset' in best_config:
                dataset_config = best_config['dataset']
                if 'horizon_idx_rel' in dataset_config:
                    print(f"  dataset.horizon_idx_rel: {dataset_config['horizon_idx_rel']}")
                if 'diff_idx_rel' in dataset_config:
                    print(f"  dataset.diff_idx_rel: {dataset_config['diff_idx_rel']}")
                if 'token_idx_rel' in dataset_config:
                    print(f"  dataset.token_idx_rel: {dataset_config['token_idx_rel']}")
            
            if 'model' in best_config:
                model_config = best_config['model']
                print(f"  model.lr: {model_config.get('lr', 'N/A')}")
                print(f"  model.lambda_reg: {model_config.get('lambda_reg', 'N/A')}")
                print(f"  model.lr_gamma: {model_config.get('lr_gamma', 'N/A')}")
            print("=" * 60)
            
            print("\nMetrics:")
            for metric in args.metrics:
                metric_short = metric.split('/')[-1]
                mean_val = best_result[f'{metric_short}_mean']
                std_val = best_result[f'{metric_short}_std']
                print(f"  {metric_short}: {mean_val:.4f} ± {std_val:.4f}")
            
            print("\nFull Configuration:")
            print(json.dumps(best_config, indent=2, default=str))
            print("=" * 120)
            
        except Exception as e:
            print(f"Error processing sweep '{sweep_name}': {e}")
            continue
    
    # Print summary of all sweeps
    if len(all_sweep_results) > 1:
        print("\n" + "=" * 120)
        print("SUMMARY: BEST CONFIGURATIONS ACROSS ALL SWEEPS")
        print("=" * 120)
        
        for sweep_name, results in all_sweep_results.items():
            print(f"\n{sweep_name}:")
            best_result = next(r for r in results['all_results'] 
                             if r['config_signature'] == results['best_config_sig'])
            
            # Print key metrics
            for metric in args.metrics:
                metric_short = metric.split('/')[-1]
                mean_val = best_result[f'{metric_short}_mean']
                std_val = best_result[f'{metric_short}_std']
                print(f"  {metric_short}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Print key hyperparameters
            best_config = results['best_config']
            if 'model' in best_config:
                model_config = best_config['model']
                print(f"  lr: {model_config.get('lr', 'N/A')}, "
                      f"lambda_reg: {model_config.get('lambda_reg', 'N/A')}, "
                      f"lr_gamma: {model_config.get('lr_gamma', 'N/A')}")
        
        print("=" * 120)
    
    # Save results to CSV if requested
    if args.output and all_sweep_results:
        # If multiple sweeps, create separate files for each
        for sweep_name, results in all_sweep_results.items():
            # Sanitize sweep name for filename
            safe_sweep_name = sweep_name.replace('/', '_').replace(' ', '_')
            
            if len(all_sweep_results) > 1:
                output_file = args.output.replace('.csv', f'_{safe_sweep_name}.csv')
                config_file = output_file.replace('.csv', '_best_config.json')
            else:
                output_file = args.output
                config_file = args.output.replace('.csv', '_best_config.json')
            
            # Create DataFrame with all results
            rows = []
            for result in results['all_results']:
                row = {
                    'sweep_name': sweep_name,
                    'n_seeds': result['n_seeds'],
                    'seeds': str(result['seeds']),
                    'run_names': str(result['run_names'])
                }
                
                # Add metrics
                for metric in args.metrics:
                    metric_short = metric.split('/')[-1]
                    row[f'{metric_short}_mean'] = result.get(f'{metric_short}_mean', float('nan'))
                    row[f'{metric_short}_std'] = result.get(f'{metric_short}_std', float('nan'))
                    row[f'{metric_short}_n'] = result.get(f'{metric_short}_n', 0)
                
                # Add flattened config
                config = result['config']
                if 'model' in config:
                    for key, value in config['model'].items():
                        row[f'model.{key}'] = value
                if 'dataset' in config:
                    for key, value in config['dataset'].items():
                        if not isinstance(value, (dict, list)):
                            row[f'dataset.{key}'] = value
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Sort by primary metric
            primary_metric_short = args.primary_metric.split('/')[-1]
            df = df.sort_values(f'{primary_metric_short}_mean', ascending=not args.maximize)
            
            df.to_csv(output_file, index=False)
            print(f"\nSaved results for '{sweep_name}' to: {output_file}")
            
            # Also save best config as JSON
            with open(config_file, 'w') as f:
                json.dump(results['best_config'], f, indent=2, default=str)
            print(f"Saved best configuration for '{sweep_name}' to: {config_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best configuration from W&B sweep runs by averaging over seeds"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="tdqc-sweeps",
        help="W&B project name (without username prefix)",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Name prefix of sweep runs to analyze (e.g., 'pizero-default-lstm-lstm_sweep'). "
             "Use this for single sweep or --sweep-names for multiple sweeps.",
    )
    parser.add_argument(
        "--sweep-names",
        type=str,
        nargs="+",
        default=None,
        help="List of sweep name prefixes to analyze in one run (e.g., 'sweep1' 'sweep2' 'sweep3')",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "falert_early_roc_auc/model_val_seen",
            "falert_early_roc_auc/model_val_unseen",
            "calibration/model_brier_at_stop_val_seen",
            "calibration/model_brier_at_stop_val_unseen",
        ],
        help="Metrics to extract and compare",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="falert_early_roc_auc/model_val_unseen",
        help="Primary metric to use for ranking configurations",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=True,
        help="Maximize the primary metric (use --no-maximize to minimize)",
    )
    parser.add_argument(
        "--no-maximize",
        dest="maximize",
        action="store_false",
        help="Minimize the primary metric instead of maximizing",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="Minimum number of seeds required for a configuration to be considered",
    )
    parser.add_argument(
        "--exclude-keys",
        type=str,
        nargs="+",
        default=['seed', 'exp_name', 'exp_suffix', 'wandb_group_name', 'run_id', 'wandb_run_id', 'run_name', 'timestamp'],
        help="Config keys to exclude when grouping configurations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path for results (optional). "
             "When processing multiple sweeps, separate files will be created for each.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output to see config signatures and differences",
    )
    
    args = parser.parse_args()
    
    # Validate that at least one of sweep-name or sweep-names is provided
    if not args.sweep_name and not args.sweep_names:
        parser.error("Either --sweep-name or --sweep-names must be provided")
    
    # If sweep-name is provided but sweep-names is not, use sweep-name
    if args.sweep_name and not args.sweep_names:
        args.sweep_names = [args.sweep_name]
    
    main(args)
