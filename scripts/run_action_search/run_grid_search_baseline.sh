#!/bin/bash
# filepath: run_grid_search.sh

# Experiment configurations
declare -a N_CANDIDATES=( 10)
declare -a TEMPERATURES=(1.5)

CUDA_DEVICE=7
ALPHA=0.2
SEED=20

count=0

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python openvla/experiments/robot/libero/run_libero_eval_with_qnetwork.py \
            --model_family openvla \
            --load_in_8bit False \
            --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
            --task_suite_name libero_10 \
            --center_crop True \
            --output_hidden_states True \
            --run_id_note "baseline" \
            --calibration_seed ${SEED} \
            --conformal_alpha ${ALPHA} \
            --save_videos True \
            --save_root "./rollouts_baseline" \
            --use_cp_guided_selection False \
            --use_conformal_prediction False 
