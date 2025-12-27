#!/bin/bash

################################################################################
# Test 8 Different Reward Functions in Parallel
# 
# Runs 8 experiments simultaneously:
# - 4 GPUs (4, 5, 6, 7)
# - 2 experiments per GPU
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ==============================================================================
# SHARED HYPERPARAMETERS
# ==============================================================================

ROAD_CLASS="D"
TOTAL_TIMESTEPS=10000000
CHECKPOINT_FREQ=4000000
LEARNING_RATE=3e-4
BUFFER_SIZE=100000
BATCH_SIZE=256
EVAL_FREQ=10000

# ==============================================================================
# GPU ASSIGNMENT
# ==============================================================================

# GPU 4: Reward 1, 2
# GPU 5: Reward 3, 4
# GPU 6: Reward 5, 6
# GPU 7: Reward 7, 8

GPUS=(4 4 5 5 6 6 7 7)
REWARD_IDS=(1 2 3 4 5 6 7 8)

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

LOG_DIR="${PROJECT_ROOT}/experiments/logs"
MODEL_DIR="${PROJECT_ROOT}/experiments/models"
RESULT_DIR="${PROJECT_ROOT}/experiments/results"
PLOT_DIR="${PROJECT_ROOT}/experiments/plots"

mkdir -p "${LOG_DIR}" "${MODEL_DIR}" "${RESULT_DIR}" "${PLOT_DIR}"

# ==============================================================================
# LAUNCH EXPERIMENTS
# ==============================================================================

echo "================================================================================"
echo "TESTING 8 REWARD FUNCTIONS IN PARALLEL"
echo "================================================================================"
echo "Road class:      ${ROAD_CLASS}"
echo "Timesteps:       ${TOTAL_TIMESTEPS}"
echo "GPUs:            4, 5, 6, 7 (2 experiments per GPU)"
echo "================================================================================"

PIDS=()
EXP_NAMES=()

for i in {0..7}; do
    REWARD_ID=${REWARD_IDS[$i]}
    GPU=${GPUS[$i]}
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME="reward${REWARD_ID}_road${ROAD_CLASS}_${TIMESTAMP}"
    EXP_NAMES+=("${EXP_NAME}")
    
    echo ""
    echo "Launching Reward ${REWARD_ID} on GPU ${GPU}..."
    
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python ${PROJECT_ROOT}/src/train_sac.py"
    CMD="${CMD} --road_class ${ROAD_CLASS}"
    CMD="${CMD} --learning_rate ${LEARNING_RATE}"
    CMD="${CMD} --buffer_size ${BUFFER_SIZE}"
    CMD="${CMD} --batch_size ${BATCH_SIZE}"
    CMD="${CMD} --total_timesteps ${TOTAL_TIMESTEPS}"
    CMD="${CMD} --eval_freq ${EVAL_FREQ}"
    CMD="${CMD} --checkpoint_freq ${CHECKPOINT_FREQ}"
    CMD="${CMD} --reward_id ${REWARD_ID}"
    CMD="${CMD} --gpu ${GPU}"
    CMD="${CMD} --log_dir ${LOG_DIR}"
    CMD="${CMD} --model_dir ${MODEL_DIR}"
    CMD="${CMD} --result_dir ${RESULT_DIR}"
    CMD="${CMD} --exp_name ${EXP_NAME}"
    
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    ERR_FILE="${LOG_DIR}/${EXP_NAME}.err"
    
    # Launch in background
    eval ${CMD} > ${LOG_FILE} 2> ${ERR_FILE} &
    PID=$!
    PIDS+=($PID)
    
    echo "  → Reward ${REWARD_ID} started (PID: ${PID}, GPU: ${GPU})"
    echo "  → Logs: ${LOG_FILE}"
    
    # Small delay to stagger starts
    sleep 2
done

echo ""
echo "================================================================================"
echo "ALL 8 EXPERIMENTS LAUNCHED"
echo "================================================================================"
echo "Process IDs: ${PIDS[@]}"
echo ""
echo "Monitor progress:"
for i in {0..7}; do
    echo "  Reward ${REWARD_IDS[$i]}: tail -f ${LOG_DIR}/${EXP_NAMES[$i]}.log"
done
echo ""
echo "Waiting for all experiments to complete..."
echo "================================================================================"

# ==============================================================================
# WAIT FOR COMPLETION
# ==============================================================================

FAILED=0

for i in {0..7}; do
    PID=${PIDS[$i]}
    REWARD_ID=${REWARD_IDS[$i]}
    
    wait ${PID}
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Reward ${REWARD_ID} completed successfully"
    else
        echo "✗ Reward ${REWARD_ID} failed with exit code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "================================================================================"

if [ ${FAILED} -eq 0 ]; then
    echo "✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
else
    echo "✗ ${FAILED} EXPERIMENTS FAILED"
fi

echo "================================================================================"

# ==============================================================================
# POST-PROCESSING: EVALUATE ALL MODELS
# ==============================================================================

if [ ${FAILED} -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "EVALUATING ALL MODELS"
    echo "================================================================================"
    
    for i in {0..7}; do
        REWARD_ID=${REWARD_IDS[$i]}
        GPU=${GPUS[$i]}
        EXP_NAME=${EXP_NAMES[$i]}
        
        FINAL_MODEL="${MODEL_DIR}/${EXP_NAME}/sac_final.zip"
        
        if [ -f "${FINAL_MODEL}" ]; then
            echo "Evaluating Reward ${REWARD_ID}..."
            
            EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU} python ${PROJECT_ROOT}/src/evaluate.py"
            EVAL_CMD="${EVAL_CMD} --model_path ${FINAL_MODEL}"
            EVAL_CMD="${EVAL_CMD} --road_class ${ROAD_CLASS}"
            EVAL_CMD="${EVAL_CMD} --reward_id ${REWARD_ID}"
            EVAL_CMD="${EVAL_CMD} --n_episodes 10"
            EVAL_CMD="${EVAL_CMD} --seed 42"
            EVAL_CMD="${EVAL_CMD} --output_dir ${RESULT_DIR}"
            EVAL_CMD="${EVAL_CMD} --exp_name ${EXP_NAME}"
            
            eval ${EVAL_CMD} > /dev/null 2>&1
            echo "  ✓ Done"
        fi
    done
    
    # ==============================================================================
    # GENERATE COMPARISON PLOT
    # ==============================================================================
    
    echo ""
    echo "================================================================================"
    echo "GENERATING REWARD COMPARISON PLOT"
    echo "================================================================================"
    
    # Create Python script to generate comparison
    COMPARISON_SCRIPT="${PROJECT_ROOT}/scripts/compare_rewards.py"
    
    cat > ${COMPARISON_SCRIPT} << 'EOF'
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from utils.plotting import SuspensionPlotter
from utils.rewards import RewardFunctions

# Load all results
result_dir = Path(sys.argv[1])
plot_dir = Path(sys.argv[2])
exp_names = sys.argv[3:]

reward_ids = []
rms_values = []
descriptions = []

for exp_name in exp_names:
    # Extract reward ID from exp_name
    reward_id = int(exp_name.split('reward')[1].split('_')[0])
    
    # Load evaluation results
    eval_file = result_dir / exp_name / 'evaluation_results.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        rms = data['aggregated_metrics']['rms_acceleration_mean']
        desc = RewardFunctions.get_reward_description(reward_id)
        
        reward_ids.append(reward_id)
        rms_values.append(rms)
        descriptions.append(desc)
        
        print(f"Reward {reward_id}: RMS = {rms:.6f} m/s² ({desc})")

# Generate plot
if reward_ids:
    plotter = SuspensionPlotter(plot_dir)
    plotter.plot_reward_function_comparison(
        reward_ids, rms_values, descriptions,
        filename='reward_comparison_all.png'
    )
    print(f"\nComparison plot saved to: {plot_dir}/reward_comparison_all.png")
else:
    print("No results found")
EOF
    
    python ${COMPARISON_SCRIPT} ${RESULT_DIR} ${PLOT_DIR} ${EXP_NAMES[@]}
    
    echo "✓ Comparison plot generated"
fi

echo ""
echo "================================================================================"
echo "REWARD FUNCTION TESTING COMPLETED"
echo "================================================================================"
echo "Results:  ${RESULT_DIR}/"
echo "Plots:    ${PLOT_DIR}/"
echo "Models:   ${MODEL_DIR}/"
echo "================================================================================"
