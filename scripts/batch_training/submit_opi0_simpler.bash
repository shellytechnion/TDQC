#!/bin/bash
# ==============================================================================
# Open-Pi-Zero on SimplerEnv
# ==============================================================================
# Model:   Open-Pi-Zero (diffusion policy; uses denoising trajectory features)
# Dataset: SimplerEnv — bridge and fractal task suites (both run in loop)
# Env var: TDQC_OPENPIZERO_ROLLOUT_ROOT
# W&B:     group=opi0_simpler_v1
#
# ALL RUNS IN THIS FILE:
#
# FINAL RUNS — all active:                     [*] = active (uncommented)
#   [*] lstm               — LSTM, cumsum loss (sweep over lr, λ, horizon/diff idx)
#   [*] mlp                — MLP,  cumsum loss (sweep over lr, λ, horizon/diff idx)
#   [*] embed (cosine/euclid) — embedding-distance baseline
#   [*] embed (mahala)        — embedding-distance baseline, Mahalanobis distance
#   [*] embed (pca_kmeans)    — embedding-distance baseline, PCA + k-means
#   [*] rnd                   — Chen et al. RND baseline
#   [*] logpzo                — Chen et al. log-probability baseline
#   [*] handcrafted           — hand-crafted metrics (no model training)
#   [*] handcrafted_multi     — hand-crafted metrics on multi-sample rollouts
#
# Note: All runs loop over datasets: open_pizero_simpler_bridge, open_pizero_simpler_fractal
# ==============================================================================

GROUP_NAME=opi0_simpler_v1

# ==============================================================================
# FINAL RUNS
# ==============================================================================

# lstm — LSTM, cumsum loss; sweep over lr, λ, horizon_idx_rel, diff_idx_rel
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=lstm \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=0-1-2 \
        train.exp_suffix=lstm

    # mlp — MLP, cumsum loss; sweep over lr, λ, horizon_idx_rel, diff_idx_rel
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=indep \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=0-1-2 \
        train.exp_suffix=mlp
done

# embed — embedding-distance baselines (cosine + Euclidean distances)
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.n_epochs=1 \
        model.distance=cosine,euclid \
        model.use_success_only=False \
        model.topk=1,5,10 \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed

    # embed — embedding-distance baseline (Mahalanobis distance)
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.n_epochs=1 \
        model.distance=mahala \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

# embed — embedding-distance baseline (PCA + k-means)
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.distance=pca_kmeans \
        model.pca_dim=32,64,128 \
        model.n_clusters=16,32,64 \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

# rnd / logpzo — Chen et al. baselines
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    # rnd — Chen et al. RND baseline
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.load_to_cuda=False \
        model=rnd \
        train.roc_every=50 \
        model.batch_size=32 \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=chen

    # logpzo — Chen et al. log-probability baseline
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.load_to_cuda=False \
        model=logpzo \
        train.roc_every=50 \
        model.batch_size=32 \
        model.forward_chunk_size=512 \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=chen
done

# handcrafted — hand-crafted metrics, no model training
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted
done

# handcrafted_multi — hand-crafted metrics on multi-sample rollouts
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${TDQC_OPENPIZERO_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted_multi
done
