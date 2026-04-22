#!/bin/bash
# filepath: run_grid_search.sh

# Experiment configurations
declare -a N_CANDIDATES=(10)
declare -a TEMPERATURES=(1.5)
declare -a QVALUE_THRESH=(0.3 0.35 0.4)

CUDA_DEVICE=7
ALPHA=0.2
SEED=20

count=0
total=$((${#N_CANDIDATES[@]} * ${#TEMPERATURES[@]}))

echo "Running grid search: ${total} experiments"

for n in "${N_CANDIDATES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
    	for thresh in "${QVALUE_THRESH[@]}"; do
	((count++))
	echo ""
	echo "[$count/$total] n_candidates=${n}, temperature=${temp}, threshold=${thresh}"
	CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python openvla/experiments/robot/libero/run_libero_eval_with_qnetwork.py \
	    --model_family openvla \
	    --load_in_8bit False \
	    --seed 9 \
	    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
	    --task_suite_name libero_10 \
	    --center_crop True \
	    --output_hidden_states True \
	    --run_id_note "grid_n${n}_temp${temp}_thresh${thresh}_seed9" \
	    --calibration_seed ${SEED} \
	    --conformal_alpha ${ALPHA} \
	    --save_videos True \
	    --save_root "./rollouts_grid_n${n}_temp${temp}_thresh${thresh}_seed9" \
	    --n_action_candidates ${n} \
	    --temperature ${temp} \
	    --use_cp_guided_selection True \
	    --q_value_thresh ${thresh}
	done
    done
done 
