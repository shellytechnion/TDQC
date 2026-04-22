#!/bin/bash
# ==============================================================================
# OpenVLA on LIBERO Suite 10 — Q-Learning model
# ==============================================================================
# Model:   Q-learning with GRU head (predicts failure probability as Q-value)
# Dataset: LIBERO suite 10
# Env var: TDQC_OPENVLA_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=openvla_libero_v2
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — best HPs, seeds 0-20:           [*] = active (uncommented)
#   [*] top_k_probs_BCE  — Q-learning GRU, BCE loss,   top-k token probs (lr=1e-5, λ=1e-3, horizon=10)
#   [*] top_k_probs_TD0  — Q-learning GRU, TD(0) loss, top-k token probs (lr=1e-5, λ=0,   horizon=1)
#   [ ] mlp_clean                        — MLP,  cumsum loss, clean baseline (no augmentation)
#   [ ] lstm_clean                       — LSTM, cumsum loss, clean baseline (no augmentation)
#
#
# SWEEPS — HP search, seeds 0-1:
#   [ ] top_k_probs_BCE_sweep  — sweep lr, λ, γ, head_hidden, gru_hidden
#   [ ] top_k_probs_TD0_sweep  — sweep lr, λ, γ, head_hidden, gru_hidden
# ==============================================================================
# ABLATION FINAL RUNS — superseded configs, kept for reference:
#   [ ] probs_only_TD0                   — Q-learning GRU, TD(0) loss, full token probability vector
#   [ ] probs_only_TDLambda              — Q-learning GRU, TD(λ) loss, full token probability vector
#   [ ] top_k_probs_only_TD0_categorical — Q-learning GRU, categorical TD(0) loss, top-k probs
#   [ ] probs_features                   — Q-learning GRU, TD(0) loss, probability features (not top-k)
#   [ ] lstm_TD0                         — LSTM, TD(0) loss (non-Q-learning baseline)
# ==============================================================================

GROUP_NAME=openvla_libero_v2

for SUITE_NAME in 10; do
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do

# ==============================================================================
# FINAL RUNS
# ==============================================================================

    # top_k_probs_BCE — Q-learning GRU, BCE loss, top-k token probs (best HP from sweep)
    python -m failure_prob.train \
         --multirun \
         train.wandb_group_name=${GROUP_NAME} \
         train.wandb_project=tdqc-final \
         dataset=openvla_libero_${SUITE_NAME} \
         dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
         dataset.token_idx_rel=1.0 \
         dataset.load_to_cuda=True \
         model=q_learning \
         model.batch_size=512 \
         model.lr=1e-5 \
         model.lambda_reg=1e-3 \
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
         model.td_horizon=10 \
         model.lr_gamma=0.1 \
         model.lr_step_size=200

    # top_k_probs_TD0 — Q-learning GRU, TD(0) loss, top-k token probs (best HP from sweep)
    python -m failure_prob.train \
         --multirun \
         train.wandb_group_name=${GROUP_NAME} \
         train.wandb_project=tdqc-final \
         dataset=openvla_libero_${SUITE_NAME} \
         dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
         dataset.token_idx_rel=1.0 \
         dataset.load_to_cuda=True \
         model=q_learning \
         model.batch_size=512 \
         model.lr=1e-5 \
         model.lambda_reg=0 \
         model.use_actions=False \
         model.loss=TDLoss \
         model.use_time_weighting=False \
         model.head_hidden=512 \
         model.gru_hidden=256 \
         model.num_gru_layers=1 \
         model.n_history_steps=-1 \
         model.n_epochs=1000 \
         train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
         train.exp_suffix=top_k_probs_TD0 \
         model.optimizer=adamw \
         model.use_top_k_probs=True \
         model.use_probs_features=False \
         model.td_horizon=1 \
         model.lr_gamma=0.8 \
         model.lr_step_size=200



    # lstm_TD0 — LSTM, TD(0) loss (non-Q-learning baseline)
    #     python -m failure_prob.train \
    #         --multirun \
    #         train.wandb_group_name=${GROUP_NAME} \
    #         dataset=openvla_libero_${SUITE_NAME} \
    #         dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #         train.wandb_project=tdqc \
    #         dataset.token_idx_rel=1.0 \
    #         dataset.load_to_cuda=False \
    #         model=lstm \
    #         model.batch_size=64 \
    #         model.lr=1e-5 \
    #         model.lambda_reg=0 \
    #         model.loss=TDLoss \
    #         model.n_epochs=1000 \
    #         model.lr_gamma=0.5 \
    #         model.lr_step_size=400 \
    #         train.seed=${SEED} \
    #         train.exp_suffix=lstm_TD0

    # mlp_clean — MLP, cumsum loss, clean baseline 
    #    python -m failure_prob.train \
    #        --multirun \
    #        train.wandb_group_name=${GROUP_NAME} \
    #        dataset=openvla_libero_${SUITE_NAME} \
    #        dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #        train.wandb_project=tdqc \
    #        dataset.token_idx_rel=1.0 \
    #        dataset.load_to_cuda=False \
    #        model=indep \
    #        model.batch_size=64 \
    #        model.lr=1e-4 \
    #        model.lambda_reg=1e-2 \
    #        train.seed=${SEED} \
    #        train.exp_suffix=mlp_clean

    # lstm_clean — LSTM, cumsum loss, clean baseline 
    #     python -m failure_prob.train \
    #         --multirun \
    #         train.wandb_group_name=${GROUP_NAME} \
    #         dataset=openvla_libero_${SUITE_NAME} \
    #         dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #         train.wandb_project=tdqc \
    #         dataset.token_idx_rel=1.0 \
    #         dataset.load_to_cuda=False \
    #         model=lstm \
    #         model.batch_size=64 \
    #         model.lr=1e-4 \
    #         model.lambda_reg=1 \
    #         train.seed=${SEED} \
    #         train.exp_suffix=lstm_clean

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # top_k_probs_BCE_sweep — Q-learning GRU BCE: sweep lr, λ, γ, head_hidden, gru_hidden
#     python -m failure_prob.train \
#          --multirun \
#          train.wandb_group_name=${GROUP_NAME} \
#          train.wandb_project=tdqc-sweeps \
#          dataset=openvla_libero_${SUITE_NAME} \
#          dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
#          dataset.token_idx_rel=1.0 \
#          dataset.load_to_cuda=True \
#          model=q_learning \
#          model.batch_size=512 \
#          model.lr=1e-5,5e-5,1e-4,1e-3 \
#          model.lambda_reg=0,1e-3,1e-1 \
#          model.use_actions=False \
#          model.loss=BCELoss \
#          model.use_time_weighting=False \
#          model.head_hidden=256,512 \
#          model.gru_hidden=256,512 \
#          model.num_gru_layers=1 \
#          model.n_history_steps=-1 \
#          model.n_epochs=1000 \
#          train.seed=0-1 \
#          train.exp_suffix=top_k_probs_BCE_sweep \
#          model.optimizer=adamw \
#          model.use_top_k_probs=True \
#          model.use_probs_features=False \
#          model.td_horizon=10 \
#          model.lr_gamma=0.8,0.1 \
#          model.lr_step_size=200

    # top_k_probs_TD0_sweep — Q-learning GRU TD(0): sweep lr, λ, γ, head_hidden, gru_hidden
#     python -m failure_prob.train \
#          --multirun \
#          train.wandb_group_name=${GROUP_NAME} \
#          train.wandb_project=tdqc-sweeps \
#          dataset=openvla_libero_${SUITE_NAME} \
#          dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
#          dataset.token_idx_rel=1.0 \
#          dataset.load_to_cuda=True \
#          model=q_learning \
#          model.batch_size=512 \
#          model.lr=1e-5,5e-5,1e-4,1e-3 \
#          model.lambda_reg=0,1e-3,1e-1 \
#          model.use_actions=False \
#          model.loss=TDLoss \
#          model.use_time_weighting=False \
#          model.head_hidden=256,512 \
#          model.gru_hidden=256,512 \
#          model.num_gru_layers=1 \
#          model.n_history_steps=-1 \
#          model.n_epochs=1000 \
#          train.seed=0-1 \
#          train.exp_suffix=top_k_probs_TD0_sweep \
#          model.optimizer=adamw \
#          model.use_top_k_probs=True \
#          model.use_probs_features=False \
#          model.td_horizon=1 \
#          model.lr_gamma=0.8,0.1 \
#          model.lr_step_size=200

# ==============================================================================
# ABLATION RUNS
# ==============================================================================

    # probs_only_TD0 — Q-learning GRU, TD(0) loss, full token probability vector (not top-k)
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc \
    #      dataset=openvla_libero_${SUITE_NAME} \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=False \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=0.00001 \
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
    #      train.exp_suffix=probs_only_TD0 \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=False \
    #      model.use_probs_features=False \
    #      model.td_horizon=10 \
    #      model.lr_gamma=0.8 \
    #      model.lr_step_size=200

    # probs_only_TDLambda — Q-learning GRU, TD(λ) loss, full token probability vector
    #  python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc \
    #      dataset=openvla_libero_${SUITE_NAME} \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=False \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=0.00001 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=TDLambdaLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=1024 \
    #      model.gru_hidden=256 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=${SEED} \
    #      train.exp_suffix=probs_only_TDLambda \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=False \
    #      model.use_probs_features=False \
    #      model.td_horizon=10 \
    #      model.lr_gamma=0.8 \
    #      model.td_lambda=0.8 \
    #      model.lr_step_size=200

    # top_k_probs_only_TD0_categorical — Q-learning GRU, categorical TD(0), top-k probs
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc \
    #      dataset=openvla_libero_${SUITE_NAME} \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=False \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=0.00001 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=CategoricalTDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=1024 \
    #      model.gru_hidden=256 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=${SEED} \
    #      train.exp_suffix=top_k_probs_only_TD0_categorical \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=True \
    #      model.use_probs_features=False \
    #      model.use_categorical=True \
    #      model.td_horizon=10 \
    #      model.lr_gamma=0.8 \
    #      model.td_lambda=0.8 \
    #      model.lr_step_size=200 \
    #      model.num_atoms=21 \
    #      model.v_min=0.0 \
    #      model.v_max=1.0 \
    #      model.sigma_ratio=0.75

    # probs_features — Q-learning GRU, TD(0), probability feature vectors (not top-k)
    # python -m failure_prob.train \
    #      --multirun \
    #      train.wandb_group_name=${GROUP_NAME} \
    #      train.wandb_project=tdqc \
    #      dataset=openvla_libero_${SUITE_NAME} \
    #      dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #      dataset.token_idx_rel=1.0 \
    #      dataset.load_to_cuda=False \
    #      model=q_learning \
    #      model.batch_size=512 \
    #      model.lr=0.00001 \
    #      model.lambda_reg=0 \
    #      model.use_actions=False \
    #      model.loss=TDLoss \
    #      model.use_time_weighting=False \
    #      model.head_hidden=1024 \
    #      model.gru_hidden=512 \
    #      model.num_gru_layers=1 \
    #      model.n_history_steps=-1 \
    #      model.n_epochs=1000 \
    #      train.seed=${SEED} \
    #      train.exp_suffix=probs_features \
    #      model.optimizer=adamw \
    #      model.use_top_k_probs=False \
    #      model.use_probs_features=True \
    #      model.use_categorical=False \
    #      model.lr_gamma=0.8 \
    #      model.lr_step_size=200

done
done
