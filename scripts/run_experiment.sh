#!/bin/bash

################################################################################
# SAC Active Suspension - Main Experiment Runner
# 
# This script trains, evaluates, and visualizes SAC for active suspension
# All hyperparameters can be configured below
################################################################################

set -e  # Exit on error

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# Experiment name (auto-generated if empty)
EXP_NAME=""

# GPU Configuration (use GPUs 4-7)
GPU=4

# ==============================================================================
# ENVIRONMENT PARAMETERS
# ==============================================================================

# Road class: A (very good) to H (extremely bad)
# Default: D (poor)
ROAD_CLASS="D"

# Vehicle speed (m/s)
VEHICLE_SPEED=20.0

# Time step (seconds)
DT=0.01

# Maximum steps per episode
MAX_STEPS=1500

# ==============================================================================
# SAC HYPERPARAMETERS
# ==============================================================================

# Learning rate
LEARNING_RATE=3e-4

# Replay buffer size
BUFFER_SIZE=100000

# Steps before training starts
LEARNING_STARTS=1000

# Batch size for training
BATCH_SIZE=256

# Soft update coefficient (tau)
TAU=0.005

# Discount factor (gamma)
GAMMA=0.99

# Training frequency (steps)
TRAIN_FREQ=1

# Gradient steps per update
GRADIENT_STEPS=1

# Entropy coefficient ('auto' or float)
ENT_COEF="auto"

# Target entropy ('auto' or float)
TARGET_ENTROPY="auto"

# ==============================================================================
# NETWORK ARCHITECTURE
# ==============================================================================

# Hidden layer dimension
HIDDEN_DIM=256

# Number of hidden layers
N_LAYERS=2

# ==============================================================================
# TRAINING PARAMETERS
# ==============================================================================

# Total training timesteps (default: 1e7)
TOTAL_TIMESTEPS=10000000

# Evaluation frequency (timesteps)
EVAL_FREQ=10000

# Checkpoint frequency (timesteps)
# Default: 4e6 (checkpoint every 4 million steps)
CHECKPOINT_FREQ=4000000

# Random seed (leave empty for random)
SEED=""

# ==============================================================================
# REWARD FUNCTION
# ==============================================================================

# Reward function ID (1-8)
# 1: Balanced (paper baseline)
# 2: Comfort-focused
# 3: Road tracking
# 4: Energy-efficient
# 5: Tire grip
# 6: Velocity damping
# 7: Suspension deflection
# 8: Exponential comfort
REWARD_ID=1

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/experiments/logs"
MODEL_DIR="${PROJECT_ROOT}/experiments/models"
RESULT_DIR="${PROJECT_ROOT}/experiments/results"
PLOT_DIR="${PROJECT_ROOT}/experiments/plots"

# ==============================================================================
# EXECUTION
# ==============================================================================

echo "================================================================================"
echo "SAC ACTIVE SUSPENSION TRAINING"
echo "================================================================================"
echo "Project root:       ${PROJECT_ROOT}"
echo "GPU:                ${GPU}"
echo "Road class:         ${ROAD_CLASS}"
echo "Total timesteps:    ${TOTAL_TIMESTEPS}"
echo "Reward function:    ${REWARD_ID}"
echo "Learning rate:      ${LEARNING_RATE}"
echo "Buffer size:        ${BUFFER_SIZE}"
echo "Batch size:         ${BATCH_SIZE}"
echo "================================================================================"

# Generate experiment name if not provided
if [ -z "$EXP_NAME" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME="sac_road${ROAD_CLASS}_reward${REWARD_ID}_${TIMESTAMP}"
    echo "Generated experiment name: ${EXP_NAME}"
fi

# Create output directories
mkdir -p "${LOG_DIR}" "${MODEL_DIR}" "${RESULT_DIR}" "${PLOT_DIR}"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU} python3 ${PROJECT_ROOT}/src/train_sac.py"
CMD="${CMD} --road_class ${ROAD_CLASS}"
CMD="${CMD} --vehicle_speed ${VEHICLE_SPEED}"
CMD="${CMD} --dt ${DT}"
CMD="${CMD} --max_steps ${MAX_STEPS}"
CMD="${CMD} --learning_rate ${LEARNING_RATE}"
CMD="${CMD} --buffer_size ${BUFFER_SIZE}"
CMD="${CMD} --learning_starts ${LEARNING_STARTS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --tau ${TAU}"
CMD="${CMD} --gamma ${GAMMA}"
CMD="${CMD} --train_freq ${TRAIN_FREQ}"
CMD="${CMD} --gradient_steps ${GRADIENT_STEPS}"
CMD="${CMD} --ent_coef ${ENT_COEF}"
CMD="${CMD} --target_entropy ${TARGET_ENTROPY}"
CMD="${CMD} --hidden_dim ${HIDDEN_DIM}"
CMD="${CMD} --n_layers ${N_LAYERS}"
CMD="${CMD} --total_timesteps ${TOTAL_TIMESTEPS}"
CMD="${CMD} --eval_freq ${EVAL_FREQ}"
CMD="${CMD} --checkpoint_freq ${CHECKPOINT_FREQ}"
CMD="${CMD} --reward_id ${REWARD_ID}"
CMD="${CMD} --gpu ${GPU}"
CMD="${CMD} --log_dir ${LOG_DIR}"
CMD="${CMD} --model_dir ${MODEL_DIR}"
CMD="${CMD} --result_dir ${RESULT_DIR}"
CMD="${CMD} --exp_name ${EXP_NAME}"

if [ -n "$SEED" ]; then
    CMD="${CMD} --seed ${SEED}"
fi

# Redirect stdout and stderr
LOG_FILE="${LOG_DIR}/${EXP_NAME}_run.log"
ERR_FILE="${LOG_DIR}/${EXP_NAME}_run.err"

echo ""
echo "Starting training..."
echo "Logs: ${LOG_FILE}"
echo "Errors: ${ERR_FILE}"
echo ""

# Run training
eval ${CMD} > ${LOG_FILE} 2> ${ERR_FILE} &
TRAIN_PID=$!

echo "Training started with PID: ${TRAIN_PID}"
echo "Monitor progress: tail -f ${LOG_FILE}"
echo ""

# Wait for training to complete
wait ${TRAIN_PID}
TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code ${TRAIN_EXIT_CODE}"
    echo "Check error log: ${ERR_FILE}"
    exit ${TRAIN_EXIT_CODE}
fi

# ==============================================================================
# EVALUATION
# ==============================================================================

echo ""
echo "================================================================================"
echo "EVALUATION"
echo "================================================================================"

# Find final model
FINAL_MODEL="${MODEL_DIR}/${EXP_NAME}/sac_final.zip"

if [ ! -f "${FINAL_MODEL}" ]; then
    echo "✗ Final model not found: ${FINAL_MODEL}"
    exit 1
fi

echo "Evaluating model: ${FINAL_MODEL}"

EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU} python ${PROJECT_ROOT}/src/evaluate.py"
EVAL_CMD="${EVAL_CMD} --model_path ${FINAL_MODEL}"
EVAL_CMD="${EVAL_CMD} --road_class ${ROAD_CLASS}"
EVAL_CMD="${EVAL_CMD} --reward_id ${REWARD_ID}"
EVAL_CMD="${EVAL_CMD} --n_episodes 10"
EVAL_CMD="${EVAL_CMD} --seed 42"
EVAL_CMD="${EVAL_CMD} --output_dir ${RESULT_DIR}"
EVAL_CMD="${EVAL_CMD} --exp_name ${EXP_NAME}"

eval ${EVAL_CMD}

echo "✓ Evaluation completed"

# ==============================================================================
# COMPARISON WITH PASSIVE
# ==============================================================================

echo ""
echo "================================================================================"
echo "COMPARISON WITH PASSIVE SUSPENSION"
echo "================================================================================"

COMPARE_CMD="CUDA_VISIBLE_DEVICES=${GPU} python ${PROJECT_ROOT}/src/compare_passive.py"
COMPARE_CMD="${COMPARE_CMD} --model_path ${FINAL_MODEL}"
COMPARE_CMD="${COMPARE_CMD} --road_class ${ROAD_CLASS}"
COMPARE_CMD="${COMPARE_CMD} --reward_id ${REWARD_ID}"
COMPARE_CMD="${COMPARE_CMD} --seed 42"
COMPARE_CMD="${COMPARE_CMD} --output_dir ${RESULT_DIR}"
COMPARE_CMD="${COMPARE_CMD} --exp_name ${EXP_NAME}"

eval ${COMPARE_CMD}

echo "✓ Comparison completed"

# ==============================================================================
# VISUALIZATION
# ==============================================================================

echo ""
echo "================================================================================"
echo "GENERATING PLOTS"
echo "================================================================================"

VIZ_CMD="python ${PROJECT_ROOT}/src/visualize.py"
VIZ_CMD="${VIZ_CMD} --results_dir ${RESULT_DIR}/${EXP_NAME}"
VIZ_CMD="${VIZ_CMD} --output_dir ${PLOT_DIR}/${EXP_NAME}"
VIZ_CMD="${VIZ_CMD} --exp_name ${EXP_NAME}"

eval ${VIZ_CMD}

echo "✓ Plots generated"
echo ""
echo "================================================================================"
echo "EXPERIMENT COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Results directory: ${RESULT_DIR}/${EXP_NAME}"
echo "Plots directory:   ${PLOT_DIR}/${EXP_NAME}"
echo "Model directory:   ${MODEL_DIR}/${EXP_NAME}"
echo "================================================================================"
