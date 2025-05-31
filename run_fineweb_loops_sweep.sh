#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Base configuration file
BASE_CONFIG="config/train_fineweb.py"
NEW_BATCH_SIZE=32
MAX_ITERS_SWEEP=5000

# n_layer is 6 as per config/train_fineweb.py
# Default max_loops is 30 as per config/train_fineweb.py

echo "Starting Looping Configurations Sweep for nanoGPT on FineWeb..."
echo "Base config: $BASE_CONFIG"
echo "Overriding batch_size to: $NEW_BATCH_SIZE"
echo "Overriding max_iters and lr_decay_iters to: $MAX_ITERS_SWEEP"
echo "--- Remember to run this script from the root of the nanoGPT directory ---"

# Experiment 1: Baseline from config file
# Uses loop_groups = [[2, 3], [4]] and max_loops = 30 from train_fineweb.py
RUN_NAME_BASE="baseline_L6_N30_lg_23_4_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (1/9): $RUN_NAME_BASE"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_BASE" \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_BASE"
echo "--------------------------------------------------"

# Experiment 2: Single Inner Group (layer 2 out of 0-5)
RUN_NAME_SINGLE_MIDDLE="lg_single_middle_2_L6_N30_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (2/9): $RUN_NAME_SINGLE_MIDDLE (loop_groups=\"[[2]]\", max_loops=30)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_SINGLE_MIDDLE" \
    --loop_groups="[[2]]" \
    --max_loops=30 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_SINGLE_MIDDLE"
echo "--------------------------------------------------"

# Experiment 3: Single Early Group (layer 0 out of 0-5)
RUN_NAME_SINGLE_EARLY="lg_single_early_0_L6_N30_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (3/9): $RUN_NAME_SINGLE_EARLY (loop_groups=\"[[0]]\", max_loops=30)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_SINGLE_EARLY" \
    --loop_groups="[[0]]" \
    --max_loops=30 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_SINGLE_EARLY"
echo "--------------------------------------------------"

# Experiment 4: Single Late Group (layer 5 out of 0-5)
RUN_NAME_SINGLE_LATE="lg_single_late_5_L6_N30_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (4/9): $RUN_NAME_SINGLE_LATE (loop_groups=\"[[5]]\", max_loops=30)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_SINGLE_LATE" \
    --loop_groups="[[5]]" \
    --max_loops=30 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_SINGLE_LATE"
echo "--------------------------------------------------"

# Experiment 5: Looping at Extrema (First [0] and Last [5] Layers, separate groups)
RUN_NAME_EXTREMA="lg_extrema_0_5_L6_N30_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (5/9): $RUN_NAME_EXTREMA (loop_groups=\"[[0],[5]]\", max_loops=30)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_EXTREMA" \
    --loop_groups="[[0],[5]]" \
    --max_loops=30 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_EXTREMA"
echo "--------------------------------------------------"

# Experiment 6: Looping every layer independently, with lower max_loops (N=5)
RUN_NAME_ALL_INDEP_LOW_N="lg_all_independent_L6_N5_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (6/9): $RUN_NAME_ALL_INDEP_LOW_N (loop_groups=\"[[0],[1],[2],[3],[4],[5]]\", max_loops=5)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_ALL_INDEP_LOW_N" \
    --loop_groups="[[0],[1],[2],[3],[4],[5]]" \
    --max_loops=5 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_ALL_INDEP_LOW_N"
echo "--------------------------------------------------"

# Experiment 7: One big looped group (all layers 0-5), default max_loops (N=30)
RUN_NAME_ALL_TOGETHER_DEFAULT_N="lg_all_together_L6_N30_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (7/9): $RUN_NAME_ALL_TOGETHER_DEFAULT_N (loop_groups=\"[[0,1,2,3,4,5]]\", max_loops=30)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_ALL_TOGETHER_DEFAULT_N" \
    --loop_groups="[[0,1,2,3,4,5]]" \
    --max_loops=30 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_ALL_TOGETHER_DEFAULT_N"
echo "--------------------------------------------------"

# Experiment 8: One big looped group (all layers 0-5), higher max_loops (N=50)
RUN_NAME_ALL_TOGETHER_HIGH_N_EXP8="lg_all_together_L6_N50_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (8/9): $RUN_NAME_ALL_TOGETHER_HIGH_N_EXP8 (loop_groups=\"[[0,1,2,3,4,5]]\", max_loops=50)"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_ALL_TOGETHER_HIGH_N_EXP8" \
    --loop_groups="[[0,1,2,3,4,5]]" \
    --max_loops=50 \
    --batch_size=$NEW_BATCH_SIZE \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_ALL_TOGETHER_HIGH_N_EXP8"
echo "--------------------------------------------------"

# Experiment 9: baseline model
RUN_NAME_BASELINE_MODEL_EXP="baseline_model_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
echo "Running Experiment (9/9): $RUN_NAME_BASELINE_MODEL_EXP baseline model"
python train.py "$BASE_CONFIG" \
    --wandb_run_name="$RUN_NAME_BASELINE_MODEL_EXP" \
    --batch_size=$NEW_BATCH_SIZE \
    --use_baseline_model=True \
    --max_iters=$MAX_ITERS_SWEEP \
    --lr_decay_iters=$MAX_ITERS_SWEEP
echo "Finished Experiment: $RUN_NAME_BASELINE_MODEL_EXP"
echo "--------------------------------------------------"

echo "Looping Configurations Sweep Finished."
