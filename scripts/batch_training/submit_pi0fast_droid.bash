#!/bin/bash
# ==============================================================================
# Pi0-FAST on DROID (real-world Franka)
# ==============================================================================
# Model:   Pi0-FAST (uses pre_logits features from fast diffusion policy)
# Dataset: DROID real-world Franka manipulation (13 tasks)
# Env var: TDQC_OPENPI_DROID_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=pizero_fast_droid_0510
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS -- best HPs, seeds 0-20:           [*] = active (uncommented)
#   [ ] mlp_BCE              -- MLP,  BCE loss,   pre_logits, token mean
#   [ ] lstm_TD0             -- LSTM, TD(0) loss, pre_logits, token mean
#   [ ] mlp_TD0              -- MLP,  TD(0) loss, pre_logits, token mean
#   [ ] lstm                 -- LSTM, cumsum loss, pre_logits, token mean
#   [ ] mlp                  -- MLP,  cumsum loss, pre_logits
#   [ ] lstm_top_k_probs_TD0 -- LSTM, TD(0) loss, top-k probs input (rebuttal)
#   [ ] lstm_top_k_probs_BCE -- LSTM, BCE loss,   top-k probs input (rebuttal)
#   [ ] 13task_handcrafted   -- log precomputed handcrafted metrics only (no model training)
#
# SWEEPS -- HP search, seeds 0-1:
#   [ ] mlp_BCE_sweep              -- sweep lr, lambda, gamma, hidden_dim
#   [ ] lstm_TD0_sweep             -- sweep lr, lambda, gamma, hidden_dim
#   [ ] mlp_TD0_sweep              -- sweep lr, lambda, gamma, hidden_dim
#   [ ] lstm_top_k_probs_TD0_sweep -- sweep lr, lambda, gamma, hidden_dim (rebuttal)
#   [ ] lstm_top_k_probs_BCE_sweep -- sweep lr, lambda, gamma, hidden_dim (rebuttal)
# ==============================================================================

GROUP_NAME=pizero_fast_droid_0510
DATASET=pizero_fast_droid_0510

# ==============================================================================
# FINAL RUNS -- best hyperparameters, seeds 0-20
# ==============================================================================


# ==============================================================================
# ARCHIVED FINAL RUNS
# ==============================================================================

    # mlp_BCE — MLP, BCE loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.token_idx_rel=mean \
    #    dataset.load_to_cuda=True \
    #    model=indep \
    #    model.n_layers=2 \
    #    model.hidden_dim=128 \
    #    model.lr=3e-4 \
    #    model.lambda_reg=0 \
    #    model.lr_gamma=0.1 \
    #    model.loss=regular_BCE \
    #    model.cumsum=False \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=mlp_BCE

    # lstm_TD0 — LSTM, TD(0) loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model=lstm \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=1 \
    #    model.hidden_dim=256 \
    #    model.lr=1e-3 \
    #    model.lambda_reg=0 \
    #    model.lr_gamma=0.8 \
    #    model.loss=TDLoss \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=lstm_TD0

    # mlp_TD0 — MLP, TD(0) loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model=indep \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=2 \
    #    model.hidden_dim=256 \
    #    model.lr=1e-3 \
    #    model.lambda_reg=0 \
    #    model.lr_gamma=0.8 \
    #    model.loss=TDLoss \
    #    train.seed=15-16-17-18-19-20 \
    #    model.cumsum=False \
    #    train.exp_suffix=mlp_TD0

    # lstm — LSTM, cumsum loss, pre_logits features, token mean
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.token_idx_rel=mean \
    #    dataset.load_to_cuda=True \
    #    model=lstm \
    #    model.n_layers=1 \
    #    model.hidden_dim=256 \
    #    model.lr=1e-3 \
    #    model.lambda_reg=0.01 \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=lstm

    # mlp — MLP, cumsum loss, pre_logits features
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.token_idx_rel=mean \
    #    dataset.load_to_cuda=True \
    #    model=indep \
    #    model.n_layers=2 \
    #    model.hidden_dim=256 \
    #    model.lr=3e-4 \
    #    model.lambda_reg=1e-3 \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=mlp

    # lstm_top_k_probs_TD0 — LSTM, TD(0) loss, top-k probs input (rebuttal)
    # Best HP: n_layers=1, hidden=256, lr=1e-4, λ=1e-3, γ=0.1
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model.input_type=top_k_probs \
    #    model=lstm \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=1 \
    #    model.hidden_dim=256 \
    #    model.lr=0.0001 \
    #    model.lambda_reg=1e-3 \
    #    model.lr_gamma=0.1 \
    #    model.loss=TDLoss \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=lstm_top_k_probs_TD0

    # lstm_top_k_probs_BCE — LSTM, BCE loss, top-k probs input (rebuttal)
    # Best HP: n_layers=1, hidden=256, lr=1e-5, λ=1e-1, γ=0.1
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-final \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model.input_type=top_k_probs \
    #    model=lstm \
    #    model.n_layers=1 \
    #    model.hidden_dim=256 \
    #    model.lr=1e-5 \
    #    model.lambda_reg=1e-1 \
    #    model.lr_gamma=0.1 \
    #    model.loss=BCELoss \
    #    train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #    train.exp_suffix=lstm_top_k_probs_BCE

   #  13task_handcrafted -- log precomputed handcrafted metrics only (no model training)
   #  python -m failure_prob.train \
   #      --multirun \
   #      train.wandb_group_name=${GROUP_NAME} \
   #      dataset=${DATASET} \
   #      dataset.data_path_prefix=/home/shellyfra/Projects/SAFE/openpi/rollouts/pi0fast_droid_0510_all/ \
   #      train.wandb_project=vla-safe-final \
   #      train.log_precomputed_only=True \
   #      train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
   #      train.exp_suffix=13task_handcrafted

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # mlp_BCE_sweep — MLP BCE: sweep lr, λ, γ, hidden_dim
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-sweeps \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.token_idx_rel=mean \
    #    dataset.load_to_cuda=True \
    #    model=indep \
    #    model.n_layers=2 \
    #    model.hidden_dim=128,256 \
    #    model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #    model.lambda_reg=0,1e-1,1e-3 \
    #    model.lr_gamma=0.8,0.1 \
    #    model.loss=regular_BCE \
    #    model.cumsum=False \
    #    train.seed=0-1 \
    #    train.exp_suffix=mlp_BCE_sweep

    # lstm_TD0_sweep — LSTM TD(0): sweep lr, λ, γ, hidden_dim
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-sweeps \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model=lstm \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=1 \
    #    model.hidden_dim=128,256 \
    #    model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #    model.lambda_reg=0,1e-3,1e-1 \
    #    model.lr_gamma=0.8,0.1 \
    #    model.loss=TDLoss \
    #    train.seed=0-1 \
    #    train.exp_suffix=lstm_TD0_sweep

    # mlp_TD0_sweep — MLP TD(0): sweep lr, λ, γ, hidden_dim
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-sweeps \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model=indep \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=2 \
    #    model.hidden_dim=128,256 \
    #    model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #    model.lambda_reg=0,1e-3,1e-1 \
    #    model.lr_gamma=0.8,0.1 \
    #    model.loss=TDLoss \
    #    train.seed=0-1 \
    #    model.cumsum=False \
    #    train.exp_suffix=mlp_TD0_sweep

    # lstm_top_k_probs_TD0_sweep — LSTM TD(0) top-k probs: sweep lr, λ, γ, hidden_dim (rebuttal)
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-sweeps \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model.input_type=top_k_probs \
    #    model=lstm \
    #    dataset.token_idx_rel=mean \
    #    model.n_layers=1 \
    #    model.hidden_dim=128,256 \
    #    model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #    model.lambda_reg=0,1e-3,1e-1 \
    #    model.lr_gamma=0.8,0.1 \
    #    model.loss=TDLoss \
    #    train.seed=0-1 \
    #    train.exp_suffix=lstm_top_k_probs_TD0_sweep

    # lstm_top_k_probs_BCE_sweep — LSTM BCE top-k probs: sweep lr, λ, γ, hidden_dim (rebuttal)
    # python -m failure_prob.train \
    #    --multirun \
    #    train.wandb_group_name=${GROUP_NAME} \
    #    train.wandb_project=tdqc-sweeps \
    #    dataset=${DATASET} \
    #    dataset.data_path_prefix=${TDQC_OPENPI_DROID_ROLLOUT_ROOT} \
    #    dataset.feat_name=pre_logits \
    #    dataset.load_to_cuda=True \
    #    model.input_type=top_k_probs \
    #    model=lstm \
    #    model.n_layers=1 \
    #    model.hidden_dim=128,256 \
    #    model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #    model.lambda_reg=0,1e-1,1e-3 \
    #    model.lr_gamma=0.8,0.1 \
    #    model.loss=BCELoss \
    #    train.seed=0-1 \
    #    train.exp_suffix=lstm_top_k_probs_BCE_sweep
