#!/bin/bash
# Run the remaining PEFT merge methods after merge_5_loras_parallel.sh.
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

results_path="${RESULTS_PATH:-./outputs/llama3_merge_parallel_results_20260422_235334}"
log_path="${results_path}/logs"
mkdir -p "$results_path" "$log_path"

datasets="commonsense,gsm8k,hexphi,mmlu"
weights="0.2 0.2 0.2 0.2 0.2"
density="0.1"

echo "=========================================="
echo "补跑 Llama3 剩余 5-LoRA 合并方法"
echo "=========================================="
echo "方法: cat, linear, ties"
echo "结果路径: ${results_path}"
echo "日志路径: ${log_path}"
echo "=========================================="

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
    local method_log="${log_path}/${method}_remaining_merge_gpu${gpu_id}.log"

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
    local code_log="${log_path}/${method}_remaining_humaneval_2gpu.log"

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

methods=("cat" "linear" "ties")
descriptions=("简单拼接，适用于不同任务的适配器" "加权线性组合，经典模型融合方法" "Trim, Elect Sign & Merge (TIES)")
use_densities=("no" "no" "yes")

overall_status=0

run_gpu0_queue() {
    run_merge_on_gpu 0 "cat" "简单拼接，适用于不同任务的适配器" "no"
    local status=$?
    if [ $status -ne 0 ]; then
        return $status
    fi

    run_merge_on_gpu 0 "ties" "Trim, Elect Sign & Merge (TIES)" "yes"
    return $?
}

run_gpu1_queue() {
    run_merge_on_gpu 1 "linear" "加权线性组合，经典模型融合方法" "no"
    return $?
}

echo "阶段 1: GPU0 跑 cat -> ties，GPU1 跑 linear"
run_gpu0_queue &
gpu0_pid=$!

run_gpu1_queue &
gpu1_pid=$!

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
    echo "至少一个剩余方法合并/普通评估失败，跳过 HumanEval。日志路径: ${log_path}"
    exit $overall_status
fi

echo "阶段 2: 双卡 HumanEval"
for i in "${!methods[@]}"; do
    run_code_eval "${methods[$i]}" "${descriptions[$i]}" "${use_densities[$i]}"
    status=$?
    if [ $status -ne 0 ]; then
        overall_status=$status
    fi
done

echo "剩余方法补跑完成。结果路径: ${results_path}"
echo "日志路径: ${log_path}"
exit $overall_status
