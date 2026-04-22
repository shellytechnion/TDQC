import os
import time
import hydra
from omegaconf import OmegaConf
from tqdm import trange

import torch
from torch.utils.data import DataLoader

import wandb

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import Rollout, RolloutDataset, RolloutDatasetContinuous, normalize_rollouts_hidden_states, ConsecutiveSampler, subsample_train_failures
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import MANUAL_METRICS, EVAL_TIME_QUANTILES
from failure_prob.utils.timer import Timer
from failure_prob.utils.routines import (
    eval_model_and_log,
    eval_metrics_and_log,
    eval_save_timing_plots,
)
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp, calc_mean_fail_step_with_cp
from failure_prob.utils.random import seed_everything

from failure_prob.conf import Config, process_cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if (
        not cfg.train.inference_only
        and (
            cfg.train.eval_save_logs
            or cfg.train.eval_save_video
            or cfg.train.eval_save_ckpt
            or cfg.train.eval_save_video_functional
            or cfg.train.eval_save_timing_plots
        )
    ):
        os.makedirs(cfg.train.logs_save_path, exist_ok=True)
        config_filename = f"config_{cfg.train.wandb_group_name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.model.loss}_{cfg.train.seed}.yaml"
        with open(os.path.join(cfg.train.logs_save_path, config_filename), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
            
    # Set seed for randomness in loading rollouts
    seed_everything(0)

    use_cuda = torch.cuda.is_available()
    if cfg.dataset.load_to_cuda and not use_cuda:
        print("WARNING: load_to_cuda=True but CUDA is unavailable; falling back to CPU")
        cfg.dataset.load_to_cuda = False

    if use_cuda:
        try:
            torch.cuda.init()
            torch.cuda.empty_cache()
            print(f"CUDA initialized successfully on device: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"WARNING: Failed to initialize CUDA: {e}")
            print("Falling back to CPU")
            use_cuda = False
            cfg.dataset.load_to_cuda = False

    device = torch.device("cuda:0" if use_cuda else "cpu")
            
    # Load and preprocess the rollout data
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")
    
    if cfg.dataset.normalize_hidden_states:
        # Normalize the hidden states to zero mean and unit variance
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)
    
    # Seed again for splitting rollouts
    if isinstance(cfg.train.seed, int):
        seeds = [cfg.train.seed]
    elif cfg.train.seed.isnumeric() and "-" not in cfg.train.seed:
        seeds = [int(cfg.train.seed)]
    else:
        # assert there are only integer seeds separated by "-" in cfg.train.seed
        assert all(s.isdigit() for s in cfg.train.seed.split("-")), "All seeds must be integers separated by '_'"

        # Run different seeds in the same call, to speed up 
        seeds = [int(s) for s in cfg.train.seed.split("-")]
        
    for seed in seeds:
        print(f"Running seed {seed}")
        cfg.train.seed = seed
        seed_everything(seed)
        
        # Handle inference-only mode: override config with saved config if provided
        if cfg.train.inference_only:
            if cfg.train.checkpoint_path is None:
                raise ValueError("checkpoint_path must be specified when inference_only=True")
            if not os.path.exists(cfg.train.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {cfg.train.checkpoint_path}")
            
            # Load config from checkpoint directory if inference_config_path is provided
            if cfg.train.inference_config_path is not None:
                if not os.path.exists(cfg.train.inference_config_path):
                    raise FileNotFoundError(f"Config file not found: {cfg.train.inference_config_path}")
                print(f"Loading config from {cfg.train.inference_config_path}")
                saved_cfg = OmegaConf.load(cfg.train.inference_config_path)
                # Preserve inference-specific settings from current config
                inference_only = cfg.train.inference_only
                checkpoint_path = cfg.train.checkpoint_path
                inference_config_path = cfg.train.inference_config_path
                current_seed = cfg.train.seed
                current_wandb = cfg.train.wandb_project
                current_wandb_group = cfg.train.wandb_group_name
                current_exp_name = cfg.train.exp_name
                current_save_video_functional = cfg.train.eval_save_video_functional
                
                # Merge saved config with current config (saved config takes precedence for model/dataset)
                # cfg = OmegaConf.merge(cfg, saved_cfg)
                cfg = saved_cfg.copy()  # Start with saved config
                
                # Restore inference-specific settings
                cfg.train.inference_only = inference_only
                cfg.train.checkpoint_path = checkpoint_path
                cfg.train.inference_config_path = inference_config_path
                cfg.train.wandb_project = current_wandb
                cfg.train.wandb_group_name = current_wandb_group
                cfg.train.exp_name = current_exp_name
                cfg.train.eval_save_video_functional = current_save_video_functional
                
                print("Config after merging:")
                print(OmegaConf.to_yaml(cfg))
        
        wandb.init(
            project = cfg.train.wandb_project, 
            dir = cfg.train.wandb_dir,
            name = cfg.train.exp_name,
            group = cfg.train.wandb_group_name,
            config = OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        
        rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
        
        if hasattr(cfg.dataset, 'max_rollout_steps') and cfg.dataset.max_rollout_steps is not None:
            print(f"Truncating rollouts to {cfg.dataset.max_rollout_steps} steps")
            rollouts_by_split_name = {
                k: RolloutDataset._truncate_rollouts(v, cfg.dataset.max_rollout_steps)
                for k, v in rollouts_by_split_name.items()
            }
        
        if getattr(cfg.dataset, 'failure_keep_ratio', None) is not None:
            ratio = cfg.dataset.failure_keep_ratio
            before = len(rollouts_by_split_name["train"])
            rollouts_by_split_name["train"] = subsample_train_failures(
                rollouts_by_split_name["train"], ratio, cfg.train.seed
            )
            after = len(rollouts_by_split_name["train"])
            wandb.log({"train_failures_kept_ratio": ratio, "train_size_before_filter": before, "train_size_after_filter": after})

        train_rollouts = rollouts_by_split_name["train"]
        all_rollouts_flat = [r for split in rollouts_by_split_name.values() for r in split]
        _input_type = getattr(cfg.model, 'input_type', 'features')
        _IMAGE_EMBED_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}
        if _input_type == 'probs' and train_rollouts[0].probs is not None:
            input_dim = max(r.probs.shape[-1] for r in all_rollouts_flat if r.probs is not None)
            print(f"Using probs as input, dim={input_dim}")
        elif _input_type == 'top_k_probs' and train_rollouts[0].top_10_probs is not None:
            _K = train_rollouts[0].top_10_probs.shape[-1]
            max_tokens = max(r.top_10_probs.shape[-2] for r in all_rollouts_flat if r.top_10_probs is not None)
            input_dim = max_tokens * _K  # n_tokens * K
            print(f"Using top_k_probs as input, dim={input_dim} ({max_tokens} tokens × {_K} K)")
        elif _input_type == 'images':
            _enc = getattr(cfg.model, 'image_encoder', 'resnet18')
            input_dim = _IMAGE_EMBED_DIMS.get(_enc, 512)
            print(f"Using images as input, encoder={_enc}, embed_dim={input_dim}")
        else:
            input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(f"hidden_feature: {train_rollouts[0].hidden_states.shape} {train_rollouts[0].hidden_states.dtype}")

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
                    num_workers=0)  if "train" in k else DataLoader(
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
                    shuffle="train" in k, 
                    num_workers=0)
                for k, v in dataset_by_split_name.items()
            }

        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            to_be_logged = eval_metrics_and_log(
                cfg,
                rollouts_by_split_name, 
                MANUAL_METRICS[cfg.dataset.name],
                EVAL_TIME_QUANTILES[cfg.dataset.name],
            )
            to_be_logged['epoch'] = 0
            wandb.log(to_be_logged)
        
        if cfg.train.log_precomputed_only:
            wandb.finish(quiet=True)
            continue
        
        model: BaseModel = get_model(cfg, input_dim)
        print(model)
        model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "num_params_M": num_params / 1e6,
            "num_trainable_params_M": num_trainable_params / 1e6,
        })
        
        if cfg.train.inference_only:
            print(f"Loading checkpoint from {cfg.train.checkpoint_path}")
            checkpoint = torch.load(cfg.train.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
            
            model.eval()
            print("Running inference-only evaluation...")

            if cfg.train.eval_save_video_functional:
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_functional")
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos_functional_cp(
                    cfg, model, 
                    rollouts_by_split_name,
                    dataloader_by_split_name,
                    video_save_folder,
                    alpha = cfg.train.eval_cp_alpha,
                )
            
            eval_logs = eval_model_and_log(
                cfg,
                model, 
                rollouts_by_split_name, 
                dataloader_by_split_name,
                EVAL_TIME_QUANTILES[cfg.dataset.name], 
                plot_score_curves=True,
                plot_auc_curves=True,
                log_classification_metrics=True,
            )
            eval_logs["epoch"] = 0
            wandb.log(eval_logs)
        else:
            optimizer, lr_scheduler = model.get_optimizer()

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            n_epochs = cfg.model.n_epochs
            epoch_losses = []
            train_start = time.time()
            pbar = trange(n_epochs)
            for epoch in pbar:
                to_be_logged = {"epoch": epoch + 1}
                model.train()
                iter_start = time.time()
                avg_loss = model.train_epoch(optimizer, dataloader_by_split_name["train"])
                epoch_time = time.time() - iter_start
                to_be_logged["epoch_time_sec"] = epoch_time
                train_ds = dataset_by_split_name["train"]
                to_be_logged["throughput_episodes_per_sec"] = len(train_ds) / epoch_time
                to_be_logged["throughput_samples_per_sec"] = len(dataloader_by_split_name["train"]) * cfg.model.batch_size / epoch_time
                if use_cuda:
                    to_be_logged["current_vram_mb"] = torch.cuda.memory_allocated(device) / 1024**2
                    to_be_logged["vram_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
                pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
                to_be_logged["train_loss"] = avg_loss

                if lr_scheduler is not None:
                    lr_scheduler.step()
                    to_be_logged["learning_rate"] = optimizer.param_groups[0]['lr']
                
                model.eval()
                if epoch % cfg.train.roc_every == 0 or epoch == n_epochs - 1:
                    eval_logs = eval_model_and_log(
                        cfg,
                        model, 
                        rollouts_by_split_name, 
                        dataloader_by_split_name,
                        EVAL_TIME_QUANTILES[cfg.dataset.name], 
                        plot_score_curves = epoch == n_epochs - 1,
                        plot_auc_curves = epoch == n_epochs - 1,
                        log_classification_metrics = epoch == n_epochs - 1,
                    )
                    to_be_logged.update(eval_logs)
                        
                    wandb.log(to_be_logged)

            train_metrics = {"train_wall_clock_sec": time.time() - train_start}
            if use_cuda:
                train_metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
            wandb.log(train_metrics)

        # Final evaluation (runs for both training and inference-only modes)
        stats, df, cp_bands = calc_mean_fail_step_with_cp(
                                cfg,
                                model, 
                                rollouts_by_split_name,
                                dataloader_by_split_name,
                                alpha=0.2,
                                calib_split_names=["val_seen"], 
                                test_split_names=["val_unseen"],
                            )
        if cfg.train.eval_save_video:
            for split, dataloader in dataloader_by_split_name.items():
                if split == "train":
                    continue
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_{split}_{cfg.dataset.name}_{cfg.model.name}_{cfg.model.loss}_{cfg.train.seed}")
                print("Saving videos to", os.path.abspath(video_save_folder))
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos(dataloader, model, cfg, video_save_folder)
                
        if cfg.train.eval_save_video_functional:
            video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_functional_{cfg.train.wandb_group_name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.model.loss}_{cfg.train.seed}")
            os.makedirs(video_save_folder, exist_ok=True)
            eval_save_videos_functional_cp(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                video_save_folder,
                alpha = cfg.train.eval_cp_alpha,
            )
            
        if cfg.train.eval_save_timing_plots:
            plot_save_folder = os.path.join(cfg.train.logs_save_path, f"timing_plots_{cfg.train.wandb_group_name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.model.loss}_{cfg.train.seed}")
            os.makedirs(plot_save_folder, exist_ok=True)
            eval_save_timing_plots(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                plot_save_folder,
                alpha= cfg.train.eval_cp_alpha,
            )
            
        if cfg.train.eval_save_ckpt:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            ckpt_save_path = os.path.join(cfg.train.logs_save_path, f"model_final_{cfg.train.wandb_group_name}_{cfg.dataset.name}_{cfg.model.name}_{cfg.model.loss}_{cfg.train.seed}.ckpt")
            print("Saving model checkpoint to", os.path.abspath(ckpt_save_path))
            torch.save(model.state_dict(), ckpt_save_path)
        
        wandb.finish(quiet=True)
        

if __name__ == "__main__":
    main()
