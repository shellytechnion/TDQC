#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Timing benchmarks — measure wall-clock training time per model/dataset combo
# ==============================================================================
# Purpose: Short runs (seeds 0-4) used to estimate time cost of each config.
#          NOT for producing final results — use the submit_* scripts for that.
#
# NOTE: No runs are currently active — all blocks below are archived.
#       Uncomment one block to re-run that timing measurement.
#
# ALL RUNS IN THIS FILE:
#
# OpenVLA / LIBERO benchmarks:
#   [ ] mlp_BCE_time_log             — MLP, BCE loss
#   [ ] lstm_time_log                — LSTM, cumsum loss
#   [ ] mlp_time_log                 — MLP,  cumsum loss
#   [ ] top_k_probs_BCE_time_log     — Q-learning GRU, BCE loss, top-k probs
#   [ ] top_k_probs_TD0_time_log     — Q-learning GRU, TD(0) loss, top-k probs
#   [ ] lstm_TD0_time_log            — LSTM, TD(0) loss
#   [ ] mlp_TD0_time_log             — MLP,  TD(0) loss
#
# Pi0-FAST / LIBERO benchmarks:
#   [ ] lstm_TD0_time_log            — LSTM, TD(0) loss, pre_logits
#   [ ] mlp_BCE_time_log             — MLP,  BCE loss,   pre_logits
#   [ ] lstm_top_k_probs_TD0_time_log— LSTM, TD(0) loss, top-k probs
#   [ ] lstm_top_k_probs_BCE_time_log— LSTM, BCE loss,   top-k probs
#   [ ] lstm_time_log                — LSTM, cumsum loss, encoded features
#   [ ] lstm_BCE_with_TDloss_time_log— LSTM, TD(0)+BCE hybrid, encoded features
#   [ ] mlp_TD0_time_log             — MLP,  TD(0) loss, pre_logits
# ==============================================================================

# ==============================================================================
# OpenVLA / LIBERO timing benchmarks
# ==============================================================================

    # mlp_BCE_time_log — MLP, BCE loss, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.token_idx_rel=1.0 \
    # 	dataset.load_to_cuda=True \
    # 	model=indep \
    # 	model.batch_size=64 \
    # 	model.lr=3e-4 \
    # 	model.lambda_reg=1e-1 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=mlp_BCE_time_log \
    # 	model.loss=regular_BCE \
    # 	model.cumsum=False

    # lstm_time_log — LSTM, cumsum loss, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.token_idx_rel=1.0 \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.batch_size=64 \
    # 	model.lr=1e-4 \
    # 	model.lambda_reg=1 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_time_log

    # mlp_time_log — MLP, cumsum loss, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.token_idx_rel=1.0 \
    # 	dataset.load_to_cuda=True \
    # 	model=indep \
    # 	model.batch_size=64 \
    # 	model.lr=1e-4 \
    # 	model.lambda_reg=1e-2 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=mlp_time_log

    # top_k_probs_BCE_time_log — Q-learning GRU, BCE loss, top-k probs, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	train.wandb_project=tdqc \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	dataset.token_idx_rel=1.0 \
    # 	dataset.load_to_cuda=True \
    # 	model=q_learning \
    # 	model.batch_size=512 \
    # 	model.lr=1e-5 \
    # 	model.lambda_reg=1e-3 \
    # 	model.use_actions=False \
    # 	model.loss=BCELoss \
    # 	model.use_time_weighting=False \
    # 	model.head_hidden=512 \
    # 	model.gru_hidden=256 \
    # 	model.num_gru_layers=1 \
    # 	model.n_history_steps=-1 \
    # 	model.n_epochs=1000 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=top_k_probs_BCE_time_log \
    # 	model.optimizer=adamw \
    # 	model.use_top_k_probs=True \
    # 	model.use_probs_features=False \
    # 	model.td_horizon=10 \
    # 	model.lr_gamma=0.1 \
    # 	model.lr_step_size=200

    # top_k_probs_TD0_time_log — Q-learning GRU, TD(0) loss, top-k probs, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	train.wandb_project=tdqc \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	dataset.token_idx_rel=1.0 \
    # 	dataset.load_to_cuda=True \
    # 	model=q_learning \
    # 	model.batch_size=512 \
    # 	model.lr=1e-5 \
    # 	model.lambda_reg=0 \
    # 	model.use_actions=False \
    # 	model.loss=TDLoss \
    # 	model.use_time_weighting=False \
    # 	model.head_hidden=512 \
    # 	model.gru_hidden=256 \
    # 	model.num_gru_layers=1 \
    # 	model.n_history_steps=-1 \
    # 	model.n_epochs=1000 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=top_k_probs_TD0_time_log \
    # 	model.optimizer=adamw \
    # 	model.use_top_k_probs=True \
    # 	model.use_probs_features=False \
    # 	model.td_horizon=1 \
    # 	model.lr_gamma=0.8 \
    # 	model.lr_step_size=200

    # lstm_TD0_time_log — LSTM, TD(0) loss, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.token_idx_rel=1.0 \
    # 	model.lr=1e-4 \
    # 	model.lambda_reg=0 \
    # 	model.lr_gamma=0.8 \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.batch_size=64 \
    # 	model.loss=TDLoss \
    # 	model.n_epochs=1000 \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_TD0_time_log

    # mlp_TD0_time_log — MLP, TD(0) loss, OpenVLA/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=openvla_libero_v2 \
    # 	dataset=openvla_libero_10 \
    # 	dataset.data_path_prefix=/path/to/TDQC/openvla/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.token_idx_rel=mean \
    # 	model.lr=1e-4 \
    # 	model.lambda_reg=0 \
    # 	model.lr_gamma=0.8 \
    # 	dataset.load_to_cuda=True \
    # 	model=indep \
    # 	model.batch_size=64 \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=mlp_TD0_time_log \
    # 	model.loss=TDLoss \
    # 	model.cumsum=False

# ==============================================================================
# Pi0-FAST / LIBERO timing benchmarks
# ==============================================================================

    # lstm_TD0_time_log — LSTM, TD(0) loss, pre_logits, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.feat_name=pre_logits \
    # 	dataset.token_idx_rel=mean \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.batch_size=64 \
    # 	model.lr=1e-3 \
    # 	model.lambda_reg=0 \
    # 	model.lr_gamma=0.1 \
    # 	model.loss=TDLoss \
    # 	model.cumsum=False \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_TD0_time_log

    # mlp_BCE_time_log — MLP, BCE loss, pre_logits, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.feat_name=pre_logits \
    # 	dataset.token_idx_rel=mean \
    # 	dataset.load_to_cuda=True \
    # 	model=indep \
    # 	model.batch_size=64 \
    # 	model.lr=1e-4 \
    # 	model.lambda_reg=1e-3 \
    # 	model.lr_gamma=0.8 \
    # 	model.loss=regular_BCE \
    # 	model.cumsum=False \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=mlp_BCE_time_log

    # lstm_top_k_probs_TD0_time_log — LSTM, TD(0) loss, top-k probs, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	model.input_type=top_k_probs \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.batch_size=64 \
    # 	model.lr=1e-5 \
    # 	model.lambda_reg=0 \
    # 	model.lr_gamma=0.1 \
    # 	model.loss=TDLoss \
    # 	model.cumsum=False \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_top_k_probs_TD0_time_log

    # lstm_top_k_probs_BCE_time_log — LSTM, BCE loss, top-k probs, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	model.input_type=top_k_probs \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.batch_size=64 \
    # 	model.lr=1e-5 \
    # 	model.lambda_reg=1e-1 \
    # 	model.lr_gamma=0.8 \
    # 	model.loss=BCELoss \
    # 	model.cumsum=False \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_top_k_probs_BCE_time_log

    # lstm_time_log — LSTM, cumsum loss, encoded features, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	train.wandb_project=tdqc \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	dataset.feat_name=encoded \
    # 	dataset.token_idx_rel=mean \
    # 	dataset.load_to_cuda=True \
    # 	model=lstm \
    # 	model.lr=3e-4 \
    # 	model.lambda_reg=1e-3 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_time_log

    # lstm_BCE_with_TDloss_time_log — LSTM, TD(0)+BCE hybrid, encoded features, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	train.wandb_project=tdqc \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	dataset.feat_name=encoded \
    # 	dataset.token_idx_rel=mean \
    # 	dataset.load_to_cuda=True \
    # 	model.loss=TDLoss \
    # 	model=lstm \
    # 	model.lr=3e-4 \
    # 	model.lambda_reg=1e-3 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=lstm_BCE_with_TDloss_time_log

    # mlp_TD0_time_log — MLP, TD(0) loss, pre_logits, Pi0-FAST/LIBERO
    # python -m failure_prob.train --multirun \
    # 	train.wandb_group_name=pi0fast_libero_v4 \
    # 	dataset=pizero_fast \
    # 	dataset.data_path_prefix=/path/to/TDQC/openpi/rollouts/ \
    # 	train.wandb_project=tdqc \
    # 	dataset.feat_name=pre_logits \
    # 	dataset.token_idx_rel=mean \
    # 	dataset.load_to_cuda=True \
    # 	model=indep \
    # 	model.batch_size=64 \
    # 	model.lr=1e-3 \
    # 	model.lambda_reg=0 \
    # 	model.lr_gamma=0.8 \
    # 	model.loss=TDLoss \
    # 	model.cumsum=False \
    # 	model.lr_step_size=200 \
    # 	train.seed=0-1-2-3-4 \
    # 	train.exp_suffix=mlp_TD0_time_log
