#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=./data/huggingface
export PROJECT_CACHE=./outputs
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 10000))}
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRa merging
model_name=qwen38b

# 适配器基础目录
LORA_DIR="/root/autodl-tmp/LoRI/outputs"

# 五个适配器路径（参考 lora_merge 脚本的路径结构）
adapter_path_1="${LORA_DIR}/codealpaca_qwen38b/LoRA_rank_32_alpha_64_lr_5e-5_bs_32/epoch-1"
adapter_path_2="${LORA_DIR}/commonsense_qwen38b/LoRA_rank_32_alpha_64_lr_5e-5_bs_32/epoch-1"
adapter_path_3="${LORA_DIR}/gsm8k_qwen38b/LoRA_rank_32_alpha_64_lr_5e-5_bs_32/epoch-1"
adapter_path_4="${LORA_DIR}/saferpaca_qwen38b/LoRA_rank_32_alpha_64_lr_5e-5_bs_32/epoch-1"
adapter_path_5="${LORA_DIR}/mmlu_qwen38b/LoRA_rank_32_alpha_64_lr_5e-5_bs_32/epoch-1"

# 适配器名称（与路径对应）
adapter_names="code commonsense math safety mmlu"

# 输出路径
results_path=./outputs/qwen38b_merge_results

echo "=========================================="
echo "开始合并和评估 Qwen3-8B 的 5 个适配器"
echo "=========================================="
echo "适配器列表:"
echo "  1. code:        ${adapter_path_1}"
echo "  2. commonsense: ${adapter_path_2}"
echo "  3. math (gsm):  ${adapter_path_3}"
echo "  4. safety:      ${adapter_path_4}"
echo "  5. mmlu:        ${adapter_path_5}"
echo "输出路径: ${results_path}"
echo ""
echo "合并方法: cat, linear, svd, ties, dare_ties, dare_linear"
echo "=========================================="
echo ""

# ============================================
# 方法 1: Concat Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 1: Concat Merging"
echo "描述: 简单拼接，适用于不同任务的适配器"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type cat \
        --weights 0.4 0.4 0.4 0.4 0.4

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type cat \
        --weights 0.4 0.4 0.4 0.4 0.4

echo "✓ Concat merging 完成"
echo ""

# ============================================
# 方法 2: Linear Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 2: Linear Merging"
echo "描述: 加权线性组合，经典的模型融合方法"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type linear \
        --weights 0.4 0.4 0.4 0.4 0.4

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type linear \
        --weights 0.4 0.4 0.4 0.4 0.4

echo "✓ Linear merging 完成"
echo ""

# ============================================
# 方法 3: SVD Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 3: SVD Merging"
echo "描述: 基于奇异值分解的低秩合并"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type svd \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type svd \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo "✓ SVD merging 完成"
echo ""

# ============================================
# 方法 4: TIES Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 4: TIES Merging"
echo "描述: Trim, Elect Sign & Merge (TIES)"
echo "论文: https://arxiv.org/abs/2306.01708"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type ties \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type ties \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo "✓ TIES merging 完成"
echo ""

# ============================================
# 方法 5: DARE-TIES Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 5: DARE-TIES Merging"
echo "描述: Drop And REscale + TIES"
echo "论文: https://arxiv.org/abs/2311.03099"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type dare_ties \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type dare_ties \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo "✓ DARE-TIES merging 完成"
echo ""

# ============================================
# 方法 6: DARE-Linear Merging
# ============================================
echo ""
echo "=========================================="
echo "方法 6: DARE-Linear Merging"
echo "描述: Drop And REscale + Linear"
echo "论文: https://arxiv.org/abs/2311.03099"
echo "=========================================="

python src/merge_5_loras.py \
        --datasets commonsense,gsm8k,hexphi,mmlu \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --results_path $results_path \
        --combination_type dare_linear \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo ""
echo "评估代码生成任务 (HumanEval)..."
accelerate launch bigcode/main_merge_5.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 $adapter_path_5 \
        --adapter_names $adapter_names \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type dare_linear \
        --weights 0.4 0.4 0.4 0.4 0.4 \
        --density 0.7

echo "✓ DARE-Linear merging 完成"
echo ""

echo "=========================================="
echo "所有合并和评估任务完成！"
echo "=========================================="
echo "测试的方法:"
echo "  1. Cat (Concatenation)"
echo "  2. Linear (Weighted Average)"
echo "  3. SVD (Singular Value Decomposition)"
echo "  4. TIES (Trim, Elect Sign & Merge)"
echo "  5. DARE-TIES (Drop And REscale + TIES)"
echo "  6. DARE-Linear (Drop And REscale + Linear)"
echo ""
echo "结果保存在: ${results_path}"
echo "=========================================="
