#!/usr/bin/env python3
"""
Script to evaluate functional conformal prediction on a trained model.
Loads a checkpoint and its config, then runs eval_save_videos_functional_cp.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add project root to import path when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import RolloutDataset, RolloutDatasetContinuous, normalize_rollouts_hidden_states, ConsecutiveSampler
from failure_prob.model import get_model
from failure_prob.utils.video import eval_save_videos_functional_cp
from failure_prob.utils.random import seed_everything
from failure_prob.conf import Config, process_cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate functional conformal prediction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints/model_final_TDQC_OpenVLA_LIBERO10.ckpt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "checkpoints/config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "videos_functional_cp"),
        help="Directory to save output videos"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Check if config exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    # Load configuration
    print(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    # Set struct flag to False to allow flexible field access
    OmegaConf.set_struct(cfg, False)
    cfg = process_cfg(cfg)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    seed_everything(cfg.train.seed)
    
    # Initialize CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load rollouts
    print("Loading rollouts...")
    all_rollouts = load_rollouts(cfg)
    print(f"Loaded {len(all_rollouts)} rollouts")
    
    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")
    
    if cfg.dataset.load_to_cuda and torch.cuda.is_available():
        all_rollouts = [r.to("cuda") for r in all_rollouts]
    
    if cfg.dataset.normalize_hidden_states:
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)
    
    # Split rollouts
    print("Splitting rollouts...")
    rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
    
    train_rollouts = rollouts_by_split_name["train"]
    input_dim = train_rollouts[0].hidden_states.shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    if cfg.model.name == "q_learning":
        cfg.model.td_horizon = cfg.model.td_horizon if cfg.model.loss == "TDLambdaLoss" else 1
        if cfg.model.loss not in ["TDLambdaLoss", "TDLoss", "CategoricalTDLoss"]:
            cfg.model.td_horizon = 0
        dataset_by_split_name = {
            k: RolloutDatasetContinuous(cfg, v) if "train" in k else RolloutDataset(cfg, v)
            for k, v in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            k: DataLoader(
                v, 
                sampler=ConsecutiveSampler(dataset_by_split_name[k], cfg.model.batch_size, cfg.model.td_horizon),
                num_workers=0) if "train" in k else DataLoader(
                v, 
                batch_size=cfg.model.batch_size, 
                shuffle=False, 
                num_workers=0)
            for k, v in dataset_by_split_name.items()
        }
    else:
        dataset_by_split_name = {
            k: RolloutDataset(cfg, v) 
            for k, v in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            k: DataLoader(
                v, 
                batch_size=cfg.model.batch_size, 
                shuffle=False, 
                num_workers=0)
            for k, v in dataset_by_split_name.items()
        }
    
    # Create model
    print("Creating model...")
    model = get_model(cfg, input_dim)
    print(model)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving videos to {os.path.abspath(args.output_dir)}")
    
    # Run functional CP evaluation
    print(f"Running functional conformal prediction with alpha={cfg.train.eval_cp_alpha}")
    eval_save_videos_functional_cp(
        cfg=cfg,
        model=model,
        rollouts_by_split_name=rollouts_by_split_name,
        dataloader_by_split_name=dataloader_by_split_name,
        save_folder=args.output_dir,
        alpha=cfg.train.eval_cp_alpha,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
