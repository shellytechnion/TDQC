#!/bin/bash
# ==============================================================================
# OpenVLA on WidowX — Q-Learning model
# ==============================================================================
# Model:   Q-learning with GRU head (predicts failure probability as Q-value)
# Dataset: WidowX real-robot manipulation
# Env var: TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=openvla_widowx_v2
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — best HPs, seeds 0-20:           [*] = active (uncommented)
#   [ ] top_k_probs_BCE  — Q-learning GRU, BCE loss, top-k token probs (lr=1e-3, λ=1e-3, γ=0.1)
#   [ ] top_k_probs (TD(0))  — Q-learning GRU, TD(0) loss, top-k token probs (lr=1e-5, λ=0, γ=0.8) 
#
# ABLATION FINAL RUNS — superseded configs, kept for reference:
#   [ ] widowx_probs_TD0     — Q-learning GRU, TD(0) loss, full probability vector (ablation)
#
# SWEEPS — HP search, seeds 0-1:
#   [ ] top_k_probs_BCE_sweep  — sweep lr, λ, γ, head_hidden, gru_hidden
#   [ ] top_k_probs_sweep      — sweep lr, λ, γ, head_hidden, gru_hidden (TD(0))
# ==============================================================================

GROUP_NAME=openvla_widowx_v2

# ==============================================================================
# FINAL RUNS
# ==============================================================================

    # top_k_probs_BCE — Q-learning GRU, BCE loss, top-k token probs (best HP from sweep)
    python -m failure_prob.train \
         --multirun \
         train.wandb_group_name=${GROUP_NAME} \
         train.wandb_project=tdqc-final \
         dataset=openvla_widowx \
         dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
         dataset.load_to_cuda=True \
         model=q_learning \
         model.batch_size=512 \
         model.lr=0.001 \
         model.lambda_reg=0.001 \
         model.use_actions=False \
         model.loss=BCELoss \
         model.use_time_weighting=False \
         model.head_hidden=512 \
         model.gru_hidden=256 \
         model.num_gru_layers=1 \
         model.n_history_steps=-1 \
         model.n_epochs=1000 \
         train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
         train.exp_suffix=top_k_probs_BCE \
         model.optimizer=adamw \
         model.use_top_k_probs=True \
         model.use_probs_features=False \
         model.lr_gamma=0.1 \
         model.lr_step_size=200

    # top_k_probs (TD(0)) — Q-learning GRU, TD(0) loss, top-k token probs
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc-final \
    #      dataset=openvla_widowx \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #      dataset.load_to_cuda=True \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=1e-5 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=TDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=512 \
    #      model.gru_hidden=1024 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #      train.exp_suffix=top_k_probs \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=True \
    #      model.use_probs_features=False \
    #      model.lr_gamma=0.1 \
    #      model.lr_step_size=200

# ==============================================================================
# ABLATION FINAL RUNS
# ==============================================================================

    # widowx_probs_TD0 — Q-learning GRU, TD(0) loss, full probability vector (not top-k)
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc-widowx \
    #      dataset=openvla_widowx \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=False \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=1e-5 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=TDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=1024 \
    #      model.gru_hidden=256 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=${SEED} \
    #      train.exp_suffix=widowx_probs_TD0 \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=False \
    #      model.use_probs_features=False \
    #      model.td_horizon=10 \
    #      model.lr_gamma=0.8 \
    #      model.lr_step_size=200

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # top_k_probs_BCE_sweep — Q-learning GRU BCE: sweep lr, λ, γ, head_hidden, gru_hidden
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc-sweeps \
    #      dataset=openvla_widowx \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #      dataset.load_to_cuda=True \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=1e-5,5e-5,1e-4,1e-3 \
    #      model.lambda_reg=0,1e-3,1e-1 \
    #      model.use_actions=False \
    #      model.loss=BCELoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=256,512 \
    #      model.gru_hidden=256,512,1024 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=0-1 \
    #      train.exp_suffix=top_k_probs_BCE_sweep \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=True \
    #      model.use_probs_features=False \
    #      model.lr_gamma=0.1,0.8 \
    #      model.lr_step_size=200

    # top_k_probs_sweep — Q-learning GRU TD(0): sweep lr, λ, γ, head_hidden, gru_hidden
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc-sweeps \
    #      dataset=openvla_widowx \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #      dataset.load_to_cuda=True \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=1e-5,5e-5,1e-4,1e-3 \
    #      model.lambda_reg=0,1e-3,1e-1 \
    #      model.use_actions=False \
    #      model.loss=TDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=256,512 \
    #      model.gru_hidden=256,512,1024 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=0-1 \
    #      train.exp_suffix=top_k_probs_sweep \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=True \
    #      model.use_probs_features=False \
    #      model.lr_gamma=0.1,0.8 \
    #      model.lr_step_size=200

    # top_k_probs_sweep (earlier widowx-specific) — sweep lr, head_hidden, gru_hidden
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc-widowx \
    #      dataset=openvla_widowx \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=True \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=1e-5,5e-5,1e-4 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=TDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=1024,512,256 \
    #      model.gru_hidden=256,512,256 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=0-1 \
    #      train.exp_suffix=top_k_probs_sweep \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=True \
    #      model.use_probs_features=False \
    #      model.lr_gamma=0.8 \    #      model.lr_step_size=200
