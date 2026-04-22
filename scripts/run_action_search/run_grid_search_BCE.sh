#!/bin/bash
# filepath: run_grid_search.sh

# Experiment configurations
declare -a N_CANDIDATES=(10)
declare -a TEMPERATURES=(1.5)
declare -a SEEDS=(8 9)

CUDA_DEVICE=4
ALPHA=0.2
SEED=20

count=0
total=$((${#N_CANDIDATES[@]} * ${#TEMPERATURES[@]} * ${#SEEDS[@]}))

echo "Running grid search: ${total} experiments"

for n in "${N_CANDIDATES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
    	for seed in "${SEEDS[@]}"; do
        ((count++))
        echo ""
        echo "[$count/$total] n_candidates=${n}, temperature=${temp}"
        
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python openvla/experiments/robot/libero/run_libero_eval_with_qnetwork.py \
            --model_family openvla \
            --load_in_8bit False \
            --seed ${seed} \
            --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
            --task_suite_name libero_10 \
            --center_crop True \
            --output_hidden_states True \
            --run_id_note "grid_n${n}_temp${temp}_BCE_${seed}" \
            --calibration_seed ${SEED} \
            --conformal_alpha ${ALPHA} \
            --save_videos True \
            --save_root "/mnt/pub/shellyf/SAFE/rollouts_grid_n${n}_temp${temp}_BCE_${seed}" \
            --n_action_candidates ${n} \
            --temperature ${temp} \
	    --qnetwork_config_path "./checkpoints/config_openvla_q_learning_BCELoss_20.yaml" \
	    --qnetwork_checkpoint "./checkpoints/model_final_openvla_q_learning_BCELoss_20.ckpt" \
	    --use_cp_guided_selection True
    	done
    done
done

#CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python openvla/experiments/robot/libero/run_libero_eval_with_qnetwork.py \
#            --model_family openvla \
#            --load_in_8bit False \
#            --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
#            --task_suite_name libero_10 \
#            --center_crop True \
#            --output_hidden_states True \
#            --run_id_note "baseline" \
#            --calibration_seed ${SEED} \
#            --conformal_alpha ${ALPHA} \
#            --save_videos True \
#            --save_root "./rollouts_baseline" \
#            --use_cp_guided_selection False \
#            --use_conformal_prediction False
