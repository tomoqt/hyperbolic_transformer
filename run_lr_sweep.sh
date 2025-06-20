#!/bin/bash

# Learning rate sweep script
# Runs 5 different learning rates from 1e-3 to 1e-5
# Tests both baseline model (True) and hyperbolic model (False)
# Uses 10k iterations for faster sweep

# Learning rates to test
learning_rates=(1e-3 5e-4 1e-4 5e-5 1e-5)

# Model types to test
model_types=("True" "False")

# New wandb project for the sweep
wandb_project="fineweb_lr_sweep"

echo "Starting learning rate sweep..."
echo "Learning rates: ${learning_rates[@]}"
echo "Model types: baseline=${model_types[@]}"
echo "Wandb project: $wandb_project"
echo "Max iterations: 10000"

# Loop through model types (baseline True/False)
for use_baseline in "${model_types[@]}"; do
    # Loop through learning rates
    for lr in "${learning_rates[@]}"; do
        # Create run name
        if [ "$use_baseline" = "True" ]; then
            model_name="baseline"
        else
            model_name="hyperbolic"
        fi
        
        run_name="${model_name}_lr_${lr}"
        
        echo ""
        echo "=========================================="
        echo "Starting run: $run_name"
        echo "Model: $model_name (baseline=$use_baseline)"
        echo "Learning rate: $lr"
        echo "=========================================="
        
        # Run the training with overridden parameters
        python train.py \
            config/train_fineweb_medium.py \
            --wandb_project="$wandb_project" \
            --wandb_run_name="$run_name" \
            --learning_rate=$lr \
            --max_iters=10000 \
            --lr_decay_iters=10000 \
            --use_baseline_model=$use_baseline \
            --use_muon=False \
            --eval_interval=500 \
            --log_interval=50
        
        # Check if the run completed successfully
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $run_name"
        else
            echo "❌ Failed: $run_name"
        fi
        
        # Small delay between runs
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "Learning rate sweep completed!"
echo "Total runs: $((${#learning_rates[@]} * ${#model_types[@]}))"
echo "Check results in wandb project: $wandb_project"
echo "==========================================" 