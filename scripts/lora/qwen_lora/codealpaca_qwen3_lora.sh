#!/bin/bash
# Set cache directories
export HF_HOME=./data/huggingface
export PROJECT_CACHE=./outputs
# export WANDB_MODE=offline
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 10000))}
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRA training
dataset_name=codealpaca
model=qwen38b
n_epochs=1
batch_size=32
grad_norm=1
save_every=epoch_$n_epochs
lora_rank=32
lora_alpha=64
sparsity_ratio=0.0

# Define learning rates to try - dense search in optimal range
learning_rates=(1.3e-4 1.35e-4 1.4e-4 )

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
    echo "=========================================="
    echo "Training with learning rate: $lr"
    echo "=========================================="

    exp_name="${dataset_name}_${model}/LoRA_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
    adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
    results_path="${PROJECT_CACHE}/${dataset_name}_${model}"

    python -u src/train_lora.py \
            model=$model \
            datasets=[$dataset_name] \
            exp_name=$exp_name \
            lr=$lr \
            save_every=$save_every \
            n_epochs=$n_epochs \
            batch_size=$batch_size \
            model.fsdp_policy_mp=bfloat16 \
            fsdp_port=$MASTER_PORT \
            optimizer=AdamW \
            grad_norm_strategy=even \
            max_grad_norm=$grad_norm \
            lora_rank=$lora_rank \
            lora_alpha=$lora_alpha

    # Evaluation on HumanEval
    accelerate launch bigcode/main.py \
            --model $model \
            --peft_model $adapter_path \
            --metric_output_path $results_path \
            --tasks humaneval \
            --temperature 0.2 \
            --n_samples 20 \
            --batch_size 10 \
            --sparsity_ratio $sparsity_ratio \
            --allow_code_execution

    echo "Completed training and evaluation for lr=$lr"
    echo ""
done

echo "All experiments completed!"

# # Shutdown the machine
# /usr/bin/shutdown -h now
