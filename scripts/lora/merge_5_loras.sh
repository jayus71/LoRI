#!/bin/bash
# Set cache directories (please update the paths to your own)
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

# LoRA merging
model_name=llama3

# Adapter paths aligned with ../lora_merge/scripts/merge_and_evaluate_three_methods.sh.
LORA_DIR="../lora_merge/data/lora_weights"
adapter_path_1="${LORA_DIR}/codealpaca"
adapter_path_2="${LORA_DIR}/commonsense"
adapter_path_3="${LORA_DIR}/gsm"
adapter_path_4="${LORA_DIR}/saferpaca"
adapter_path_5="${LORA_DIR}/mmlu"

# Adapter names aligned with the Qwen LoRA merge script.
adapter_names="code commonsense math safety mmlu"

# Output path
date_suffix=$(date +%Y%m%d)
results_path=./outputs/llama3_merge_results_${date_suffix}
mkdir -p "$results_path"

datasets="commonsense,gsm8k,hexphi,mmlu"
weights="0.2 0.2 0.2 0.2 0.2"
density="0.1"

echo "=========================================="
echo "开始合并和评估 Llama3 的 5 个适配器"
echo "=========================================="
echo "适配器列表:"
echo "  1. code:        ${adapter_path_1}"
echo "  2. commonsense: ${adapter_path_2}"
echo "  3. math (gsm):  ${adapter_path_3}"
echo "  4. safety:      ${adapter_path_4}"
echo "  5. mmlu:        ${adapter_path_5}"
echo "输出路径: ${results_path}"
echo ""
echo "本次测试方法（已去除 SVD 系列）: cat, linear, ties, dare_ties, dare_linear, magnitude_prune"
echo "注意: 所有使用 pruning / density 的方法统一设置为保留 10% 参数 (density=0.1)"
echo "=========================================="
echo ""

run_method() {
    local method="$1"
    local description="$2"
    local use_density="$3"

    echo ""
    echo "=========================================="
    echo "方法: ${method}"
    echo "描述: ${description}"
    echo "权重: 均等 (0.2 x 5 = 1.0)"
    if [ "$use_density" = "yes" ]; then
        echo "Density: ${density} (保留 10% 参数)"
    else
        echo "Density: N/A"
    fi
    echo "=========================================="

    local merge_cmd=(
        python src/merge_5_loras.py
        --datasets "$datasets"
        --model_name "$model_name"
        --adapter_path "$adapter_path_1" "$adapter_path_2" "$adapter_path_3" "$adapter_path_4" "$adapter_path_5"
        --adapter_names $adapter_names
        --results_path "$results_path"
        --combination_type "$method"
        --weights $weights
    )

    local eval_cmd=(
        accelerate launch bigcode/main_merge_5.py
        --model "$model_name"
        --peft_model "$adapter_path_1" "$adapter_path_2" "$adapter_path_3" "$adapter_path_4" "$adapter_path_5"
        --adapter_names $adapter_names
        --metric_output_path "$results_path"
        --tasks humaneval
        --temperature 0.2
        --n_samples 20
        --batch_size 10
        --allow_code_execution
        --combination_type "$method"
        --weights $weights
    )

    if [ "$use_density" = "yes" ]; then
        merge_cmd+=(--density "$density")
        eval_cmd+=(--density "$density")
    fi

    "${merge_cmd[@]}"
    local merge_status=$?
    if [ $merge_status -ne 0 ]; then
        echo "✗ ${method} merge 失败，跳过评估"
        return $merge_status
    fi

    echo ""
    echo "评估代码生成任务 (HumanEval)..."
    "${eval_cmd[@]}"
    local eval_status=$?
    if [ $eval_status -ne 0 ]; then
        echo "✗ ${method} HumanEval 评估失败"
        return $eval_status
    fi

    echo "✓ ${method} 完成"
    return 0
}

# run_method cat "简单拼接，适用于不同任务的适配器" no
# run_method linear "加权线性组合，经典模型融合方法" no
# run_method ties "Trim, Elect Sign & Merge (TIES)" yes
run_method dare_ties "Drop And REscale + TIES" yes
run_method dare_linear "Drop And REscale + Linear" yes
run_method magnitude_prune "按权重幅值裁剪后合并" yes

echo ""
echo "=========================================="
echo "所有合并和评估任务已执行完毕"
echo "=========================================="
echo "测试的方法及参数:"
echo "  1. cat:             weights=0.2x5, density=N/A"
echo "  2. linear:          weights=0.2x5, density=N/A"
echo "  3. ties:            weights=0.2x5, density=0.1"
echo "  4. dare_ties:       weights=0.2x5, density=0.1"
echo "  5. dare_linear:     weights=0.2x5, density=0.1"
echo "  6. magnitude_prune: weights=0.2x5, density=0.1"
echo ""
echo "注意: 所有权重已归一化（总和=1.0）"
echo "结果保存在: ${results_path}"
echo "=========================================="
