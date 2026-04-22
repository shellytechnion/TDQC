#!/usr/bin/env python3
"""
Update pickle files to replace probs and top10_probs with values computed
from hidden states through lm_head.

This ensures consistency between the saved probs/top10_probs and what would
be computed from the hidden states.

Also verifies that:
- Actions computed from hidden states match CSV actions
- Probs computed from hidden states are close to CSV probs

Usage:
    python scripts/update_pickle_probs_from_hidden_states.py \
        --rollout_dir openvla/rollouts/single-foward/openvla_widowx \
        --model_path openvla/openvla-7b
"""

import argparse
import sys
from pathlib import Path
import json
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[1] / "openvla"))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_for_hidden_states(model_path: str):
    """Load OpenVLA model for running hidden states through lm_head."""
    print("[*] Loading OpenVLA model for hidden state inference")

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load full model (needed to get lm_head)
    vla = OpenVLAForActionPrediction.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to GPU
    vla = vla.to(DEVICE)
    vla.eval()

    return vla


def compute_actions_and_probs_from_hidden_states(model, hidden_states):
    """
    Compute probs and top10_probs from hidden states using lm_head.
    
    Args:
        model: The OpenVLA model (needs lm_head)
        hidden_states: Tensor of shape (7, 4096) - one hidden state per action token
    
    Returns:
        probs: numpy array of shape (7,) - probability of chosen token for each action
        top10_probs: numpy array of shape (7, 10) - top 10 logit values for each action token
        actions: numpy array of shape (7,) - normalized actions (in [-1, 1] range)
    """
    # Ensure hidden_states is on the right device and dtype
    if isinstance(hidden_states, np.ndarray):
        hidden_states = torch.from_numpy(hidden_states)
    hidden_states = hidden_states.to(device=DEVICE, dtype=torch.bfloat16)
    
    # Run through lm_head to get logits
    with torch.no_grad():
        full_logits = model.language_model.lm_head(hidden_states)  # (7, 32064)

    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    n_action_bins = model.config.n_action_bins
    
    # Slice to action bins
    action_logits = full_logits[:, vocab_size - n_action_bins + 1 : vocab_size + 1]  # (7, 256)
    
    # Compute probs: softmax then max (matching unc_utils.py)
    token_prob = F.softmax(action_logits, dim=-1)  # (7, 256)
    probs = token_prob.max(dim=-1).values.float().cpu().numpy()  # (7,)
    
    # Compute top10_probs: top 10 logit values for each token
    top10_values, top10_indices = torch.topk(action_logits, k=10, dim=-1)  # (7, 10)
    top10_probs = top10_values.float().cpu().numpy()  # (7, 10)
    
    # Compute actions from full logits (same as predict_action)
    predicted_token_ids = torch.argmax(full_logits, dim=-1).cpu().numpy()  # (7,)
    bins = np.linspace(-1, 1, n_action_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    discretized_actions = vocab_size - predicted_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    normalized_actions = bin_centers[discretized_actions]
    
    return probs, top10_probs, normalized_actions


def unnormalize_actions(normalized_actions, model, unnorm_key="bridge_orig"):
    """Unnormalize actions using model's norm_stats."""
    action_norm_stats = model.norm_stats[unnorm_key]["action"]
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def load_csv_row(csv_path: Path, row_idx: int):
    """Load actions and probs from a specific row in CSV."""
    df = pd.read_csv(csv_path)
    if row_idx >= len(df):
        return None, None
    
    row = df.iloc[row_idx]
    
    # Get actions
    action_cols = ['action/dx', 'action/dy', 'action/dz', 'action/droll', 'action/dpitch', 'action/dyaw', 'action/dgripper']
    actions = row[action_cols].values.astype(np.float64)
    
    # Get probs
    prob_cols = [f'action/token_{i}_prob' for i in range(7)]
    probs = row[prob_cols].values.astype(np.float64)
    
    return actions, probs


def process_pickle_file(pkl_path: Path, model, unnorm_key: str, dry_run: bool = False, verify: bool = True):
    """
    Process a single pickle file, updating probs and top10_probs.
    Optionally verify against CSV that actions match and probs are close.
    
    Returns:
        dict with 'success', 'action_match', 'prob_max_diff' keys
    """
    result = {'success': False, 'action_match': None, 'prob_max_diff': None}
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  Error loading {pkl_path.name}: {e}")
        return result
    
    if 'hidden_states' not in data:
        print(f"  Skipping {pkl_path.name}: No hidden_states")
        return result
    
    hidden_states_list = data['hidden_states']
    if len(hidden_states_list) == 0:
        print(f"  Skipping {pkl_path.name}: Empty hidden_states")
        return result
    
    # Compute new probs and top10_probs for each timestep
    num_timesteps = len(hidden_states_list)
    new_probs = []
    new_top10_probs = []
    all_normalized_actions = []
    
    for t in range(num_timesteps):
        hidden_states = hidden_states_list[t]  # (7, 4096)
        probs, top10_probs, normalized_actions = compute_actions_and_probs_from_hidden_states(model, hidden_states)
        new_probs.append(probs)
        new_top10_probs.append(top10_probs)
        all_normalized_actions.append(normalized_actions)
    
    # Stack into arrays
    new_probs = np.stack(new_probs, axis=0)  # (T, 7)
    new_top10_probs = np.stack(new_top10_probs, axis=0)  # (T, 7, 10)
    all_normalized_actions = np.stack(all_normalized_actions, axis=0)  # (T, 7)
    
    # Verify against CSV if requested
    csv_path = pkl_path.with_suffix('.csv')
    action_max_diff = 0.0
    prob_max_diff = 0.0
    
    if verify and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            num_csv_rows = min(len(df), num_timesteps)
            
            for t in range(num_csv_rows):
                csv_actions, csv_probs = load_csv_row(csv_path, t)
                if csv_actions is None:
                    continue
                
                # Unnormalize our computed actions
                model_actions = unnormalize_actions(all_normalized_actions[t], model, unnorm_key)
                
                # Compare actions
                action_diff = np.abs(model_actions - csv_actions).max()
                action_max_diff = max(action_max_diff, action_diff)
                
                # Compare probs
                prob_diff = np.abs(new_probs[t] - csv_probs).max()
                prob_max_diff = max(prob_max_diff, prob_diff)
            
            result['action_match'] = action_max_diff < 0.001
            result['action_max_diff'] = action_max_diff
            result['prob_max_diff'] = prob_max_diff
            
        except Exception as e:
            print(f"  Warning: Could not verify {pkl_path.name} against CSV: {e}")
    
    if dry_run:
        # Just print what would change
        old_probs_shape = np.array(data.get('probs', [])).shape if 'probs' in data else None
        old_top10_shape = np.array(data.get('top10_probs', [])).shape if 'top10_probs' in data else None
        verify_str = ""
        if result['action_match'] is not None:
            action_status = "✓" if result['action_match'] else "✗"
            verify_str = f" | actions {action_status}, prob_diff={result['prob_max_diff']:.5f}"
        print(f"  {pkl_path.name}: probs {old_probs_shape} -> {new_probs.shape}, top10 {old_top10_shape} -> {new_top10_probs.shape}{verify_str}")
    else:
        # Update the data
        data['probs'] = new_probs
        data['top10_probs'] = new_top10_probs
        
        # Save back to pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
    
    result['success'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description="Update pickle files with probs computed from hidden states")
    parser.add_argument("--rollout_dir", type=str, required=True,
                       help="Directory containing rollout pickle files")
    parser.add_argument("--model_path", type=str, default="openvla/openvla-7b",
                       help="Path to model")
    parser.add_argument("--dry_run", action="store_true",
                       help="Don't actually modify files, just show what would change")
    parser.add_argument("--verify", action="store_true",
                       help="Verify computed actions/probs against CSV files")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig",
                       help="Key for unnormalizing actions")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model_for_hidden_states(args.model_path)
    
    # Find all pickle files
    pkl_files = list(rollout_dir.glob("**/*.pkl"))
    print(f"Found {len(pkl_files)} pickle files")
    
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
        print(f"Processing first {args.max_files} files")
    
    if args.dry_run:
        print("\n*** DRY RUN - no files will be modified ***\n")
    
    if args.verify:
        print("*** VERIFICATION ENABLED - checking against CSV files ***\n")
    
    # Process each pickle file
    success_count = 0
    fail_count = 0
    action_matches = 0
    action_mismatches = 0
    max_action_diffs = []
    max_prob_diffs = []
    
    for pkl_path in tqdm(pkl_files, desc="Processing pickle files"):
        result = process_pickle_file(pkl_path, model, args.unnorm_key, 
                                     dry_run=args.dry_run, verify=args.verify)
        if result['success']:
            success_count += 1
            if result['action_match'] is not None:
                if result['action_match']:
                    action_matches += 1
                else:
                    action_mismatches += 1
                max_action_diffs.append(result['action_max_diff'])
                max_prob_diffs.append(result['prob_max_diff'])
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files found: {len(pkl_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed/Skipped: {fail_count}")
    
    if args.verify and max_prob_diffs:
        print(f"\nVerification Results:")
        print(f"  Actions match CSV: {action_matches}/{action_matches + action_mismatches}")
        print(f"  Action mismatches: {action_mismatches}")
        print(f"  Max action diff (max across files): {max(max_action_diffs):.6f}")
        print(f"  Mean action diff (mean of max per file): {np.mean(max_action_diffs):.6f}")
        print(f"  Max prob diff (max across files): {max(max_prob_diffs):.6f}")
        print(f"  Mean prob diff (mean of max per file): {np.mean(max_prob_diffs):.6f}")
    
    if args.dry_run:
        print("\n*** DRY RUN - no files were modified ***")
    else:
        print(f"\nUpdated probs and top10_probs in {success_count} pickle files")
    print("="*60)


if __name__ == "__main__":
    main()
