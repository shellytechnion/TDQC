#!/bin/bash
# ==============================================================================
# OpenVLA on LIBERO Suite 10
# ==============================================================================
# Model:   OpenVLA (LLM backbone; uses per-step token probabilities as features)
# Dataset: LIBERO suite 10
# Env var: TDQC_OPENVLA_ROLLOUT_ROOT
# W&B:     project=tdqc-final, group=openvla_libero_v2
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — best HPs, seeds 0-20:           [*] = active (uncommented)
#   [ ] lstm_BCE_top_k_probs_GRU_best  — LSTM-GRU, BCE loss,   top-k probs (lr=1e-5, λ=1e-3, γ=0.8, hidden=256)
#   [ ] lstm_TD0_top_k_probs_GRU_best  — LSTM-GRU, TD(0) loss, top-k probs (lr=3e-4, λ=0, γ=0.8, hidden=128)
#   [ ] lstm_TD0_top_k_probs_2000   — LSTM, TD(0) loss, top-k probs, 2000 epochs (rebuttal)
#   [ ] lstm_images_res18           — LSTM, BCE loss, raw image input via ResNet-18 encoder (rebuttal)
#   [ ] lstm_TD0                    — LSTM, TD(0) loss, single token at rel pos 1.0
#   [ ] lstm                        — LSTM, BCE loss
#   [ ] mlp_TD0                     — MLP,  TD(0) loss, token mean
#   [ ] mlp_BCE                     — MLP,  BCE loss,   token at rel pos 1.0
#   [ ] mlp                         — MLP,  cumsum loss
#   [ ] handcrafted                 — hand-crafted token-level metrics (no model)
#
# SWEEPS — HP search, seeds 0-1:
#   [ ] lstm_images_res18_sweep           — sweep over lr, λ, γ
#   [ ] lstm_BCE_top_k_probs_GRU_sweep    — sweep over lr, λ, γ (GRU variant)
#   [ ] lstm_TD0_top_k_probs_GRU_sweep    — sweep over lr, λ, γ (GRU + TD(0))
#   [ ] mlp_BCE_sweep                     — sweep over lr, λ, token_idx_rel
#   [ ] mlp_TD0_sweep                     — sweep over lr, λ, γ, token_idx_rel
#   [ ] lstm_TD0_sweep                    — sweep over lr, λ, γ, token_idx_rel
# ==============================================================================

GROUP_NAME=openvla_libero_v2

for SUITE_NAME in 10; do

# ==============================================================================
# FINAL RUNS
# ==============================================================================

    # lstm_BCE_top_k_probs_GRU_best — LSTM-GRU, BCE loss, top-k probs (best HP from sweep)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     model.lr=1e-5 \
    #     model.lambda_reg=1e-3 \
    #     model.lr_gamma=0.8 \
    #     model.hidden_dim=256 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=top_k_probs \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_BCE_top_k_probs_GRU_best

    # lstm_TD0_top_k_probs_GRU_best — LSTM-GRU, TD(0) loss, top-k probs (best HP from sweep)
    # displayName: openvla-10-lstm-lstm_TD0_top_k_probs_GRU_sweep1, model.lr: 0.0003, model.lr_gamma: 0.8, model.hidden_dim: 128, model.lambda_reg: 0
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
        train.wandb_project=tdqc-final \
        model.lr=0.0003 \
        model.lambda_reg=0 \
        model.lr_gamma=0.8 \
        model.hidden_dim=128 \
        dataset.load_to_cuda=True \
        model.input_type=top_k_probs \
        model.loss=TDLoss \
        model=lstm \
        model.batch_size=64 \
        model.n_epochs=1000 \
        model.lr_step_size=200 \
        model.use_gru=True \
        train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
        train.exp_suffix=lstm_TD0_top_k_probs_GRU_best

# ==============================================================================
# ARCHIVED FINAL RUNS
# ==============================================================================

    # mlp_BCE — MLP, BCE loss, token at rel pos 1.0
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.token_idx_rel=1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=3e-4 \
    #     model.lambda_reg=1e-1 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_BCE \
    #     model.loss=regular_BCE \
    #     model.cumsum=False

    # mlp_TD0 — MLP, TD(0) loss, token mean
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.token_idx_rel=mean \
    #     model.lr=1e-4 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.8 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp_TD0 \
    #     model.loss=TDLoss \
    #     model.cumsum=False

    # lstm_TD0 — LSTM, TD(0) loss, token at rel pos 1.0
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.token_idx_rel=1.0 \
    #     model.lr=1e-4 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.8 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.loss=TDLoss \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_TD0

    # lstm — LSTM, cumsum loss
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.token_idx_rel=1.0 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.lr=1e-4 \
    #     model.lambda_reg=1 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm

    # mlp — MLP, cumsum loss
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     dataset.token_idx_rel=1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-4 \
    #     model.lambda_reg=1e-2 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=mlp

    # lstm_images_res18 — LSTM, BCE loss, raw image input via ResNet-18 encoder (rebuttal)
    # Best HP from sweep: lr=5e-5, λ=1e-1, γ=0.8
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     model.lr=5e-5 \
    #     model.lambda_reg=1e-1 \
    #     model.lr_gamma=0.8 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=images \
    #     model.image_encoder=resnet18 \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_images_res18

    # lstm_TD0_top_k_probs_2000 — LSTM, TD(0) loss, top-k probs, 2000 epochs (rebuttal)
    # Best HP from sweep: lr=5e-5, λ=0, γ=0.8, hidden=256
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-final \
    #     model.lr=5e-5 \
    #     model.lambda_reg=0 \
    #     model.lr_gamma=0.8 \
    #     model.hidden_dim=256 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=top_k_probs \
    #     model.loss=TDLoss \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=2000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20 \
    #     train.exp_suffix=lstm_TD0_top_k_probs_2000

    # handcrafted — hand-crafted token-level metrics, no model training
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.log_precomputed_only=True \
    #     train.seed=0-1-2 \
    #     train.exp_suffix=handcrafted

# ==============================================================================
# SWEEPS — hyperparameter search (seeds 0-1, multi-value grids)
# ==============================================================================

    # mlp_BCE_sweep — MLP BCE: sweep lr ∈ {1e-5…1e-3}, λ ∈ {0,1e-3,1e-1}, token_idx_rel ∈ {mean,0,1}
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_BCE_sweep \
    #     model.loss=regular_BCE \
    #     model.cumsum=False

    # mlp_TD0_sweep — MLP TD(0): sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     dataset.load_to_cuda=True \
    #     model=indep \
    #     model.batch_size=64 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=mlp_TD0_sweep \
    #     model.loss=TDLoss \
    #     model.cumsum=False

    # lstm_TD0_sweep — LSTM TD(0): sweep lr, λ, γ, token_idx_rel
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     dataset.token_idx_rel=mean,0.0,1.0 \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     dataset.load_to_cuda=True \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.loss=TDLoss \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_TD0_sweep

    # lstm_images_res18_sweep — LSTM image (ResNet-18): sweep lr, λ, γ (rebuttal)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=images \
    #     model.image_encoder=resnet18 \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_images_res18_sweep

    # lstm_BCE_top_k_probs_GRU_sweep — LSTM-GRU BCE top-k probs: sweep lr, λ, γ (rebuttal)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.hidden_dim=256 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=top_k_probs \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     model.use_gru=True \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_BCE_top_k_probs_GRU_sweep

    # lstm_TD0_top_k_probs_GRU_sweep — LSTM-GRU TD(0) top-k probs: sweep lr, λ, γ (rebuttal)
    # python -m failure_prob.train \
    #     --multirun \
    #     train.wandb_group_name=${GROUP_NAME} \
    #     dataset=openvla_libero_${SUITE_NAME} \
    #     dataset.data_path_prefix=${TDQC_OPENVLA_ROLLOUT_ROOT} \
    #     train.wandb_project=tdqc-sweeps \
    #     model.lr=1e-5,5e-5,1e-4,3e-4,1e-3 \
    #     model.lambda_reg=0,1e-1,1e-3 \
    #     model.lr_gamma=0.8,0.1 \
    #     model.hidden_dim=256,128 \
    #     dataset.load_to_cuda=True \
    #     model.input_type=top_k_probs \
    #     model.loss=TDLoss \
    #     model=lstm \
    #     model.batch_size=64 \
    #     model.n_epochs=1000 \
    #     model.lr_step_size=200 \
    #     model.use_gru=True \
    #     train.seed=0-1 \
    #     train.exp_suffix=lstm_TD0_top_k_probs_GRU_sweep

done
