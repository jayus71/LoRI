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
date_suffix=$(date +%Y%m%d_%H%M%S)
results_path=./outputs/llama3_merge_parallel_results_${date_suffix}
mkdir -p "$results_path"
log_path="${results_path}/logs"
mkdir -p "$log_path"

datasets="commonsense,gsm8k,hexphi,mmlu"
weights="0.2 0.2 0.2 0.2 0.2"
density="0.1"

echo "=========================================="
echo "开始并行合并和评估 Llama3 的 5 个适配器"
echo "=========================================="
echo "适配器列表:"
echo "  1. code:        ${adapter_path_1}"
echo "  2. commonsense: ${adapter_path_2}"
echo "  3. math (gsm):  ${adapter_path_3}"
echo "  4. safety:      ${adapter_path_4}"
echo "  5. mmlu:        ${adapter_path_5}"
echo "输出路径: ${results_path}"
echo "日志路径: ${log_path}"
echo ""
echo "运行策略: 合并/普通评估分发到 GPU0/GPU1；全部完成后再用双卡跑 HumanEval"
echo "注意: 所有使用 pruning / density 的方法统一设置为保留 10% 参数 (density=0.1)"
echo "=========================================="
echo ""

build_merge_cmd() {
    local method="$1"
    local use_density="$2"

    MERGE_CMD=(
        python src/merge_5_loras.py
        --datasets "$datasets"
        --model_name "$model_name"
        --adapter_path "$adapter_path_1" "$adapter_path_2" "$adapter_path_3" "$adapter_path_4" "$adapter_path_5"
        --adapter_names $adapter_names
        --results_path "$results_path"
        --combination_type "$method"
        --weights $weights
    )

    if [ "$use_density" = "yes" ]; then
        MERGE_CMD+=(--density "$density")
    fi
}

run_merge_on_gpu() {
    local gpu_id="$1"
    local method="$2"
    local description="$3"
    local use_density="$4"
    local method_log="${log_path}/${method}_merge_gpu${gpu_id}.log"

    build_merge_cmd "$method" "$use_density"
    {
        echo "=========================================="
        echo "方法: ${method}"
        echo "描述: ${description}"
        echo "GPU: ${gpu_id}"
        echo "权重: 均等 (0.2 x 5 = 1.0)"
        if [ "$use_density" = "yes" ]; then
            echo "Density: ${density} (保留 10% 参数)"
        else
            echo "Density: N/A"
        fi
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES="$gpu_id" "${MERGE_CMD[@]}"
        local status=$?
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "退出状态: ${status}"
        exit $status
    } > "$method_log" 2>&1
}

run_code_eval() {
    local method="$1"
    local description="$2"
    local use_density="$3"
    local code_log="${log_path}/${method}_humaneval_2gpu.log"

    local eval_cmd=(
        accelerate launch --multi_gpu --num_processes 2 bigcode/main_merge_5.py
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
        eval_cmd+=(--density "$density")
    fi

    echo ""
    echo "=========================================="
    echo "双卡评估代码生成任务 (HumanEval): ${method}"
    echo "描述: ${description}"
    echo "日志: ${code_log}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0,1 "${eval_cmd[@]}" > "$code_log" 2>&1
    local eval_status=$?
    if [ $eval_status -ne 0 ]; then
        echo "✗ ${method} HumanEval 评估失败，查看日志: ${code_log}"
        return $eval_status
    fi

    echo "✓ ${method} HumanEval 完成"
    return 0
}

methods=("dare_ties" "dare_linear" "magnitude_prune")
descriptions=("Drop And REscale + TIES" "Drop And REscale + Linear" "按权重幅值裁剪后合并")
use_densities=("yes" "yes" "yes")

echo "=========================================="
echo "阶段 1: 合并/普通评估，按方法分发到 GPU0/GPU1"
echo "=========================================="

overall_status=0

run_gpu0_queue() {
    run_merge_on_gpu 0 "dare_ties" "Drop And REscale + TIES" "yes"
    local status=$?
    if [ $status -ne 0 ]; then
        return $status
    fi

    run_merge_on_gpu 0 "magnitude_prune" "按权重幅值裁剪后合并" "yes"
    return $?
}

run_gpu1_queue() {
    run_merge_on_gpu 1 "dare_linear" "Drop And REscale + Linear" "yes"
    return $?
}

echo "启动 GPU0 队列: dare_ties -> magnitude_prune"
run_gpu0_queue &
gpu0_pid=$!

echo "启动 GPU1 队列: dare_linear"
run_gpu1_queue &
gpu1_pid=$!

echo ""
echo "等待 GPU0/GPU1 队列完成..."
if wait "$gpu0_pid"; then
    echo "✓ GPU0 队列完成"
else
    status=$?
    echo "✗ GPU0 队列失败，状态: ${status}"
    overall_status=$status
fi

if wait "$gpu1_pid"; then
    echo "✓ GPU1 队列完成"
else
    status=$?
    echo "✗ GPU1 队列失败，状态: ${status}"
    overall_status=$status
fi

if [ $overall_status -ne 0 ]; then
    echo ""
    echo "至少一个合并/普通评估任务失败，跳过 HumanEval。日志路径: ${log_path}"
    exit $overall_status
fi

echo ""
echo "=========================================="
echo "阶段 2: 所有合并/普通评估完成，开始双卡 HumanEval"
echo "=========================================="

for i in "${!methods[@]}"; do
    run_code_eval "${methods[$i]}" "${descriptions[$i]}" "${use_densities[$i]}"
    status=$?
    if [ $status -ne 0 ]; then
        overall_status=$status
    fi
done

echo ""
echo "=========================================="
echo "所有合并和评估任务已执行完毕"
echo "=========================================="
echo "测试的方法及参数:"
echo "  1. dare_ties:       weights=0.2x5, density=0.1"
echo "  2. dare_linear:     weights=0.2x5, density=0.1"
echo "  3. magnitude_prune: weights=0.2x5, density=0.1"
echo ""
echo "注意: 所有权重已归一化（总和=1.0）"
echo "结果保存在: ${results_path}"
echo "日志保存在: ${log_path}"
echo "=========================================="

exit $overall_status
