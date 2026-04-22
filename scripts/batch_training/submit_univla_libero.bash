#!/bin/bash
# ==============================================================================
# UniVLA on LIBERO
# ==============================================================================
# Model:   UniVLA (uses hidden_states features from the VLA backbone)
# Dataset: LIBERO (univla-libero_10)
# Env var: TDQC_UNIVLA_ROLLOUT_ROOT
#          The dataset yaml sets data_path=univla-libero_10, so the full path
#          will be ${TDQC_UNIVLA_ROLLOUT_ROOT}univla-libero_10
# W&B:     project=tdqc-final, group=univla_libero_v1
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — best HPs, seeds 0-20:           [*] = active (uncommented)
#   [*] handcrafted  — log precomputed metrics only, no model training
#
# ARCHIVED FINAL RUNS — superseded configs, kept for reference:
#   [ ] lstm_top_k_probs_TD0     — LSTM, TD(0) loss, top-k probs (lr=3e-5, λ=1e-3, γ=0.1)
#   [ ] lstm_top_k_probs_BCE     — LSTM, BCE loss,   top-k probs (lr=1e-3, λ=0,    γ=0.1)
#   [ ] lstm_top_k_probs_TDLambda— LSTM, TD(λ) loss, top-k probs (lr=3e-4, λ=1e-3, γ=0.8, horizon=5) (ablation for TD(0) vs TD(λ))
#   [ ] lstm_TD0                 — LSTM, TD(0) loss, hidden_states (lr=3e-5, λ=1e-3, γ=0.1)
#   [ ] lstm_BCE                 — LSTM, BCE loss,   hidden_states (lr=1e-5, λ=1e-1, γ=0.1)
#   [ ] mlp_TD0                  — MLP,  TD(0) loss, hidden_states (lr=1e-4, λ=1e-3, γ=0.8)
#   [ ] mlp_BCE                  — MLP,  BCE loss,   hidden_states (lr=1e-5, λ=1e-1, γ=0.1)
#   [ ] mlp                      — MLP,  cumsum loss, hidden_states (lr=1e-3, λ=1e-1, γ=0.1)
#
# SWEEPS — HP search, seeds 0-1:
#   [ ] lstm_top_k_probs_TD0_sweep   — sweep lr, λ, γ, token_idx_rel
#   [ ] lstm_top_k_probs_BCE_sweep   — sweep lr, λ, γ, token_idx_rel
#   [ ] lstm_TD0_sweep               — sweep lr, λ, γ, token_idx_rel
#   [ ] lstm_BCE_sweep               — sweep lr, λ, γ
#   [ ] mlp_TD0_sweep                — sweep lr, λ, γ, token_idx_rel
#   [ ] mlp_BCE_sweep                — sweep lr, λ, γ, token_idx_rel
#   [ ] mlp_sweep                    — sweep lr, λ, γ, token_idx_rel
# ==============================================================================

GROUP_NAME=univla_libero_v1
WANDB_PROJECT=tdqc-final

# ==============================================================================
# FINAL RUNS
# ==============================================================================

    # handcrafted — log precomputed metrics only (no model training)
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        train.wandb_project=${WANDB_PROJECT} \
        dataset=univla \
        dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=handcrafted

# ==============================================================================
# ARCHIVED FINAL RUNS
# ==============================================================================

    # lstm_top_k_probs_TD0 — LSTM, TD(0) loss, top-k probs input
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=tdqc-final \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.input_type=top_k_probs \
    #     model.batch_size=64 \
    #     model.lr=0.00003 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     model.lr_step_size=200 \
    #     model.loss=TDLoss \
    #     train.exp_suffix=lstm_top_k_probs_TD0

    # lstm_top_k_probs_BCE — LSTM, BCE loss, top-k probs input
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.input_type=top_k_probs \
    #     model.batch_size=64 \
    #     model.lr=0.001 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     model.loss=BCELoss \
    #     train.exp_suffix=lstm_top_k_probs_BCE

    # lstm_top_k_probs_TDLambda — LSTM, TD(λ) loss, top-k probs input
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=tdqc-final \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.input_type=top_k_probs \
    #     model.batch_size=64 \
    #     model.lr=3e-4 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.8 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     model.lr_step_size=200 \
    #     model.loss=TDLambdaLoss \
    #     model.td_horizon=5 \
    #     train.exp_suffix=lstm_top_k_probs_TDLambda

    # lstm_TD0 — LSTM, TD(0) loss, hidden_states features
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=3e-5 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     model.loss=TDLoss \
    #     train.exp_suffix=lstm_TD0

    # lstm_BCE — LSTM, BCE loss, hidden_states features
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5 \
    #     model.lambda_reg=1e-1 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     model.loss=regular \
    #     train.exp_suffix=lstm_BCE

    # mlp_TD0 — MLP, TD(0) loss, hidden_states features
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=0.0001 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.8 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_TD0 \
    #     model.loss=TDLoss \
    #     model.cumsum=False

    # mlp_BCE — MLP, BCE loss, hidden_states features
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5 \
    #     model.lambda_reg=1e-1 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_BCE \
    #     model.loss=regular_BCE \
    #     model.cumsum=False

    # mlp — MLP, cumsum loss, hidden_states features
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=False \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=0.001 \
    #     model.lambda_reg=1e-1 \
    #     model.lr_gamma=0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp \
    #     model.cumsum=True

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # lstm_top_k_probs_TD0_sweep — LSTM TD(0) top-k probs: sweep lr, λ, γ, td_horizon
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.input_type=top_k_probs \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1 \
    #     model.lr_step_size=200 \
    #     model.loss=TDLambdaLoss \
    #     model.td_horizon=5,10 \
    #     train.exp_suffix=lstm_top_k_probs_TD0_sweep2

    # lstm_top_k_probs_BCE_sweep — LSTM BCE top-k probs: sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.input_type=top_k_probs \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     model.loss=BCELoss \
    #     train.exp_suffix=lstm_top_k_probs_BCE_sweep1

    # lstm_TD0_sweep — LSTM TD(0) hidden_states: sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     model.loss=TDLoss \
    #     train.exp_suffix=lstm_TD0

    # lstm_BCE_sweep — LSTM BCE hidden_states: sweep lr, λ, γ
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1 \
    #     model.loss=regular \
    #     train.exp_suffix=lstm_BCE

    # mlp_TD0_sweep — MLP TD(0) hidden_states: sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_TD0 \
    #     model.loss=TDLoss \
    #     model.cumsum=False

    # mlp_BCE_sweep — MLP BCE hidden_states: sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_BCE_sweep \
    #     model.loss=regular_BCE \
    #     model.cumsum=False

    # mlp_sweep — MLP cumsum hidden_states: sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     train.wandb_project=${WANDB_PROJECT} \
    #     dataset=univla \
    #     dataset.data_path_prefix=${TDQC_UNIVLA_ROLLOUT_ROOT} \
    #     dataset.feat_name=hidden_states \
    #     dataset.token_idx_rel=0.0,1.0,mean \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-3,1e-1 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.n_epochs=1000 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_sweep \
    #     model.cumsum=True
