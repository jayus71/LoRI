#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lori
export HF_HOME=./data/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROJECT_CACHE=./outputs
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 10000))}
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

set +e

model_name=llama3
LORA_DIR="../lora_merge/data/lora_weights"
adapter_path_1="${LORA_DIR}/codealpaca"
adapter_path_2="${LORA_DIR}/commonsense"
adapter_path_3="${LORA_DIR}/gsm"
adapter_path_4="${LORA_DIR}/saferpaca"
adapter_path_5="${LORA_DIR}/mmlu"
adapter_names="code commonsense math safety mmlu"

results_path=./outputs/llama3_aligned_weights_20260514
mkdir -p "$results_path"

datasets="commonsense,gsm8k,hexphi,mmlu"
# 与 sparse 推荐配置对齐：code=0.06, commonsense/mmlu=0.18, gsm8k=0.20, saferpaca=0.18
weights="0.06 0.18 0.20 0.18 0.18"

run_method() {
    local method="$1"
    echo "=========================================="
    echo "方法: ${method} | 权重: ${weights}"
    echo "=========================================="

    python src/merge_5_loras.py \
        --datasets "$datasets" \
        --model_name "$model_name" \
        --adapter_path "$adapter_path_1" "$adapter_path_2" "$adapter_path_3" "$adapter_path_4" "$adapter_path_5" \
        --adapter_names $adapter_names \
        --results_path "$results_path" \
        --combination_type "$method" \
        --weights $weights
    local s=$?
    [ $s -ne 0 ] && echo "✗ ${method} merge 失败" && return $s

    echo "评估 HumanEval..."
    accelerate launch bigcode/main_merge_5.py \
        --model "$model_name" \
        --peft_model "$adapter_path_1" "$adapter_path_2" "$adapter_path_3" "$adapter_path_4" "$adapter_path_5" \
        --adapter_names $adapter_names \
        --metric_output_path "$results_path" \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type "$method" \
        --weights $weights
    echo "✓ ${method} 完成"
}

run_method cat
run_method linear

echo "结果保存在: ${results_path}"
