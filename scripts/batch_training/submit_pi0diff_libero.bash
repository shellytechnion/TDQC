#!/bin/bash
# ==============================================================================
# Pi0-Diffusion on LIBERO
# ==============================================================================
# Model:   Pi0 diffusion policy (uses denoising trajectory features as input)
# Dataset: LIBERO (pizero dataset)
# Env var: TDQC_OPENPI_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=pi0diff_libero_v1
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS -- best HPs, seeds 0-20:           [*] = active (uncommented)
#   [*] lstm                            -- LSTM, BCE loss, best HP (lr=1e-3, lambda=1e-3, horizon=0.0, diff=1.0)
#   [*] mlp                             -- MLP,  cumsum loss, best HP (lr=3e-5, lambda=1e-3, horizon=0.0, diff=1.0)
#   [*] lstm_TD0_failure_keep_ratio_1   -- LSTM, TD(0), keep 100% of failure trajectories (rebuttal)
#   [*] lstm_TD0_failure_keep_ratio_06  -- LSTM, TD(0), keep 60%  of failure trajectories (rebuttal)
#   [*] lstm_TD0_failure_keep_ratio_03  -- LSTM, TD(0), keep 30%  of failure trajectories (rebuttal)
#   [*] lstm_TD0_failure_keep_ratio_01  -- LSTM, TD(0), keep 10%  of failure trajectories (rebuttal)
#   [ ] lstm_TD0  -- LSTM, TD(0) loss (original, no failure ratio filtering)
#   [ ] mlp_TD0   -- MLP,  TD(0) loss
#   [ ] mlp_BCE   -- MLP,  BCE loss
#
# SWEEPS -- HP search, seeds 0-1:
#   [ ] lstm_sweep      -- LSTM BCE: sweep lr, lambda, horizon_idx_rel, diff_idx_rel
#   [ ] lstm_TD0_sweep  -- LSTM TD(0): sweep lr, lambda, gamma, horizon_idx_rel, diff_idx_rel
#   [ ] mlp_TD0_sweep   -- MLP  TD(0): sweep lr, lambda, gamma, horizon_idx_rel, diff_idx_rel
#   [ ] mlp_BCE_sweep   -- MLP  BCE: sweep lr, lambda, gamma, horizon_idx_rel, diff_idx_rel
# ==============================================================================

GROUP_NAME=pi0diff_libero_v1

# ==============================================================================
# FINAL RUNS -- best hyperparameters, seeds 0-20
# ==============================================================================

    # lstm -- LSTM, cumsum loss, best HP (lr=1e-3, lambda=1e-3, horizon=0.0, diff=1.0)
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.load_to_cuda=True \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=1.0 \
        model=lstm \
        model.lr=1e-3 \
        model.lambda_reg=1e-3 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm

    # mlp -- MLP, cumsum loss, best HP (lr=3e-5, lambda=1e-3, horizon=0.0, diff=1.0)
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.load_to_cuda=True \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=1.0 \
        model=indep \
        model.lr=3e-5 \
        model.lambda_reg=1e-3 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=mlp

# ==============================================================================
# FINAL RUNS -- rebuttal ablation: vary the fraction of failure trajectories kept
# ==============================================================================

    # lstm_TD0_failure_keep_ratio_1 -- LSTM, TD(0), keep 100% of failure trajectories
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=concat-2 \
        dataset.load_to_cuda=True  \
        dataset.failure_keep_ratio=1.0 \
        model=lstm \
        model.lr=3e-4 \
        model.lambda_reg=0 \
        model.loss=TDLoss \
        model.cumsum=False \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0_failure_keep_ratio_1

    # lstm_TD0_failure_keep_ratio_06 -- LSTM, TD(0), keep 60% of failure trajectories
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=concat-2 \
        dataset.load_to_cuda=True  \
        dataset.failure_keep_ratio=0.6 \
        model=lstm \
        model.lr=3e-4 \
        model.lambda_reg=0 \
        model.loss=TDLoss \
        model.cumsum=False \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0_failure_keep_ratio_06

    # lstm_TD0_failure_keep_ratio_03 -- LSTM, TD(0), keep 30% of failure trajectories
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=concat-2 \
        dataset.load_to_cuda=True  \
        dataset.failure_keep_ratio=0.3 \
        model=lstm \
        model.lr=3e-4 \
        model.lambda_reg=0 \
        model.loss=TDLoss \
        model.cumsum=False \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0_failure_keep_ratio_03

    # lstm_TD0_failure_keep_ratio_01 -- LSTM, TD(0), keep 10% of failure trajectories
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=concat-2 \
        dataset.load_to_cuda=True  \
        dataset.failure_keep_ratio=0.1 \
        model=lstm \
        model.lr=3e-4 \
        model.lambda_reg=0 \
        model.loss=TDLoss \
        model.cumsum=False \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0_failure_keep_ratio_01

# ==============================================================================
# ARCHIVED FINAL RUNS
# ==============================================================================

    # lstm_TD0 -- LSTM, TD(0) loss, original config (no failure ratio filtering)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.horizon_idx_rel=0.0 \
    #     dataset.diff_idx_rel=concat-2 \
    #     dataset.load_to_cuda=True  \
    #     model=lstm \
    #     model.lr=3e-4 \
    #     model.lambda_reg=0 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_gamma=0.1 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_TD0

    # mlp_TD0 -- MLP, TD(0) loss
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.horizon_idx_rel=concat-2 \
    #     dataset.diff_idx_rel=1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.lr=1e-3 \
    #     model.lambda_reg=0 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_gamma=0.8 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_TD0

    # mlp_BCE -- MLP, BCE loss
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.load_to_cuda=True \
    #     dataset.horizon_idx_rel=concat-2 \
    #     dataset.diff_idx_rel=1.0 \
    #     model=indep \
    #     model.lr=3e-5 \
    #     model.lambda_reg=0 \
    #     model.loss=regular_BCE \
    #     model.cumsum=False \
    #     model.lr_gamma=0.8 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_BCE

# ==============================================================================
# SWEEPS -- hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # lstm_sweep -- LSTM BCE: sweep lr, lambda, horizon_idx_rel, diff_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.horizon_idx_rel=0.0,1.0,concat-2 \
    #     dataset.diff_idx_rel=0.0,1.0,concat-2 \
    #     model=lstm \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=1e-3,1e-2,1e-1 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_sweep

    # lstm_TD0_sweep -- LSTM TD(0): sweep lr, lambda, gamma, horizon_idx_rel, diff_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.horizon_idx_rel=0.0,1.0,concat-2 \
    #     dataset.diff_idx_rel=0.0,1.0,concat-2 \
    #     dataset.load_to_cuda=True  \
    #     model=lstm \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_gamma=0.8,0.1 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_TD0_sweep

    # mlp_TD0_sweep -- MLP TD(0): sweep lr, λ, γ, horizon_idx_rel, diff_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.horizon_idx_rel=0.0,1.0,concat-2 \
    #     dataset.diff_idx_rel=0.0,1.0,concat-2 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_gamma=0.8,0.1 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_TD0_sweep

    # mlp_BCE_sweep -- MLP BCE: sweep lr, λ, γ, horizon_idx_rel, diff_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.load_to_cuda=True \
    #     dataset.horizon_idx_rel=0.0,1.0,concat-2 \
    #     dataset.diff_idx_rel=0.0,1.0,concat-2 \
    #     model=indep \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.loss=regular_BCE \
    #     model.cumsum=False \
    #     model.lr_gamma=0.8,0.1 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_BCE_sweep
