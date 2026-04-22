#!/bin/bash
# ==============================================================================
# Pi0-FAST on LIBERO
# ==============================================================================
# Model:   Pi0-FAST (uses pre_logits / encoded features from fast diffusion policy)
# Dataset: LIBERO (pizero_fast dataset)
# Env var: TDQC_OPENPI_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=pi0fast_libero_v4
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS -- best HPs, seeds 0-20:           [*] = active (uncommented)
#   [ ] mlp                   -- MLP,  cumsum loss, pre_logits, token=1.0 (lr=1e-4, lambda=1e-2)
#   [ ] lstm_TD0              -- LSTM, TD(0) loss, pre_logits features, token mean
#   [ ] mlp_BCE               -- MLP,  BCE loss,   pre_logits features, token mean
#   [ ] mlp_TD0               -- MLP,  TD(0) loss, pre_logits features, token mean
#   [ ] lstm_top_k_probs_TD0  -- LSTM, TD(0) loss, top-k probs input (rebuttal)
#   [ ] lstm_top_k_probs_BCE  -- LSTM, BCE loss,   top-k probs input (rebuttal)
#
# SWEEPS -- HP search, seeds 0-1:
#   [ ] lstm_TD0_sweep              -- sweep lr, λ, γ, feat_name, token_idx_rel
#   [ ] mlp_BCE_sweep               -- sweep lr, λ, γ, feat_name, token_idx_rel
#   [ ] mlp_TD0_sweep               -- sweep lr, λ, γ, feat_name, token_idx_rel
#   [ ] lstm_top_k_probs_TD0_sweep  -- sweep lr, λ, γ (rebuttal)
#   [ ] lstm_top_k_probs_BCE_sweep  -- sweep lr, λ, γ (rebuttal)
# ==============================================================================

GROUP_NAME=pi0fast_libero_v4

# ==============================================================================
# FINAL RUNS -- best hyperparameters, seeds 0-20
# ==============================================================================

    # mlp -- MLP, cumsum loss, pre_logits, token=1.0 (lr=1e-4, lambda=1e-2)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=/home/shellyfra/Projects/SAFE/openpi/rollouts/ \
    #     train.wandb_project=vla-safe-final \
    #     dataset.feat_name=pre_logits \
    #     dataset.token_idx_rel=1.0 \
    #     model=indep \
    #     model.lr=1e-4 \
    #     model.lambda_reg=1e-2 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp

# ==============================================================================
# ARCHIVED FINAL RUNS
# ==============================================================================

    # lstm_TD0 -- LSTM, TD(0) loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.feat_name=pre_logits \
    #     dataset.token_idx_rel=mean \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-3 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_TD0

    # mlp_BCE -- MLP, BCE loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.feat_name=pre_logits \
    #     dataset.token_idx_rel=mean \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-4 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.8 \
    #     model.loss=regular_BCE \
    #     model.cumsum=False \
    #     train.seed=17-18-19-20 \
    #     train.exp_suffix=mlp_BCE

    # mlp_TD0 -- MLP, TD(0) loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.feat_name=pre_logits \
    #     dataset.token_idx_rel=mean \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-3 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.8 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_TD0

    # lstm_top_k_probs_TD0 -- LSTM, TD(0) loss, top-k probs input (rebuttal; best HP lr=1e-5, λ=0, γ=0.1)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     model.input_type=top_k_probs \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_top_k_probs_TD0

    # lstm_top_k_probs_BCE -- LSTM, BCE loss, top-k probs input (rebuttal; best HP lr=1e-5, λ=1e-1, γ=0.8)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     model.input_type=top_k_probs \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5 \
    #     model.lambda_reg=1e-1 \
    #     model.lr_gamma=0.8 \
    #     model.loss=BCELoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_top_k_probs_BCE

# ==============================================================================
# SWEEPS -- hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # lstm_TD0_sweep -- LSTM TD(0): sweep lr, λ, γ, feat_name, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.feat_name=pre_logits,encoded \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_TD0_sweep

    # mlp_BCE_sweep -- MLP BCE: sweep lr, λ, γ, feat_name, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.feat_name=pre_logits,encoded \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.loss=regular_BCE \
    #     model.cumsum=False \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_BCE_sweep

    # mlp_TD0_sweep -- MLP TD(0): sweep lr, λ, γ, feat_name, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.feat_name=pre_logits,encoded \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_TD0_sweep

    # lstm_top_k_probs_TD0_sweep -- LSTM TD(0) top-k probs: sweep lr, λ, γ (rebuttal)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     model.input_type=top_k_probs \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.loss=TDLoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_top_k_probs_TD0_sweep

    # lstm_top_k_probs_BCE_sweep -- LSTM BCE top-k probs: sweep lr, λ, γ (rebuttal)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=pizero_fast \
    #     dataset.data_path_prefix=${TDQC_OPENPI_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     model.input_type=top_k_probs \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.loss=BCELoss \
    #     model.cumsum=False \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_top_k_probs_BCE_sweep
