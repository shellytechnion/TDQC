#!/bin/bash
# ==============================================================================
# OpenVLA on WidowX
# ==============================================================================
# Model:   OpenVLA (LLM backbone; uses per-step token probabilities as features)
# Dataset: WidowX real-robot manipulation
# Env var: TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=openvla_widowx_v2
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — best HPs, seeds 0-20:           [*] = active (uncommented)
#   [*] lstm      — LSTM, cumsum loss                      (lr=3e-4, λ=1e-2)
#   [*] mlp       — MLP,  cumsum loss                      (lr=3e-4, λ=1e-1)
#   [*] mlp_BCE   — MLP,  BCE loss,   token at rel pos 1.0 (lr=1e-5, λ=1e-1)
#   [*] mlp_TD0   — MLP,  TD(0) loss, token at rel pos 1.0 (lr=3e-4, λ=0,   γ=0.1)
#   [*] lstm_TD0  — LSTM, TD(0) loss, token at rel pos 1.0 (lr=5e-5, λ=1e-3, γ=0.1)
#   [ ] handcrafted                              — hand-crafted metrics (no model)
#
# SWEEPS — HP search, seeds 0-1:
#   [ ] mlp_BCE_sweep   — sweep lr, λ, token_idx_rel
#   [ ] mlp_TD0_sweep   — sweep lr, λ, γ, token_idx_rel
#   [ ] lstm_TD0_sweep  — sweep lr, λ, γ, token_idx_rel
#   [ ] lstm_sweep      — sweep lr, λ, token_idx_rel
#   [ ] mlp_sweep       — sweep lr, λ, token_idx_rel
# ==============================================================================

GROUP_NAME=openvla_widowx_v2

# ==============================================================================
# FINAL RUNS
# ==============================================================================

    # lstm — LSTM, cumsum loss
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.token_idx_rel=1.0 \
        dataset.load_to_cuda=True \
        model=lstm \
        model.batch_size=64 \
        model.lr=3e-4 \
        model.lambda_reg=1e-2 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm

    # mlp — MLP, cumsum loss
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        dataset.token_idx_rel=1.0 \
        dataset.load_to_cuda=True \
        model=indep \
        model.batch_size=64 \
        model.lr=3e-4 \
        model.lambda_reg=1e-1 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=mlp

    # mlp_BCE — MLP, BCE loss, no cumsum
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        train.wandb_project=tdqc-final \
        dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        dataset.token_idx_rel=1.0 \
        dataset.load_to_cuda=True \
        model=indep \
        model.batch_size=64 \
        model.lr=1e-5 \
        model.lambda_reg=1e-1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=mlp_BCE \
        model.loss=regular_BCE \
        model.cumsum=False

    # mlp_TD0 — MLP, TD(0) loss, no cumsum
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        train.wandb_project=tdqc-final \
        dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        dataset.token_idx_rel=1.0 \
        dataset.load_to_cuda=True \
        model=indep \
        model.batch_size=64 \
        model.lr=3e-4 \
        model.lambda_reg=0 \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=mlp_TD0 \
        model.loss=TDLoss \
        model.cumsum=False

    # lstm_TD0 — LSTM, TD(0) loss
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        train.wandb_project=tdqc-final \
        dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        dataset.token_idx_rel=1.0 \
        dataset.load_to_cuda=True \
        model=lstm \
        model.batch_size=64 \
        model.lr=5e-5 \
        model.lambda_reg=1e-3 \
        model.loss=TDLoss \
        model.lr_gamma=0.1 \
        model.lr_step_size=200 \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0

    # handcrafted — hand-crafted token-level metrics, no model training
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     train.log_precomputed_only=True \
    #     train.seed=0-1-2 \
    #     train.exp_suffix=handcrafted

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # mlp_BCE_sweep — MLP BCE: sweep lr, λ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_BCE_sweep \
    #     model.loss=regular_BCE \
    #     model.cumsum=False

    # mlp_TD0_sweep — MLP TD(0): sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_TD0_sweep \
    #     model.loss=TDLoss \
    #     model.cumsum=False

    # lstm_TD0_sweep — LSTM TD(0): sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.loss=TDLoss \
    #     model.lr_gamma=0.1,0.8 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_TD0_sweep

    # lstm_sweep — LSTM cumsum: sweep lr, λ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-4,3e-4,1e-3 \
    #     model.lambda_reg=1e-3,1e-2,1e-1,1 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_sweep

    # mlp_sweep — MLP cumsum: sweep lr, λ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_widowx \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-4,3e-4,1e-3 \
    #     model.lambda_reg=1e-3,1e-2,1e-1,1 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_sweep
