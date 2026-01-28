#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=./data/huggingface
export PROJECT_CACHE=./outputs
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 10000))}
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRI-D merging
model_name=llama3
adapter_path_1=../lora_merge/data/lora_weights/commonsense
adapter_path_2=../lora_merge/data/lora_weights/gsm
adapter_path_3=../lora_merge/data/lora_weights/codealpaca
adapter_path_4=../lora_merge/data/lora_weights/saferpaca
adapter_path_5=../lora_merge/data/lora_weights/mmlu
results_path=./outputs

# Concat merging
python src/merge_5_loras.py \
        --dataset commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --results_path $results_path \
        --combination_type cat --weights 0.4 0.4 0.4 0.4 0.4

accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type cat --weights 0.4 0.4 0.4 0.4 0.4

# Linear merging
python src/merge_5_loras.py \
        --dataset commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --results_path $results_path \
        --combination_type linear --weights 0.4 0.4 0.4 0.4 0.4

accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type linear --weights 0.4 0.4 0.4 0.4 0.4
