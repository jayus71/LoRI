#!/bin/bash

set -euo pipefail

# 激活 lori 环境（如果需要）
if [[ "$CONDA_DEFAULT_ENV" != "lori" ]]; then
    echo "激活 lori 环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate lori
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Step 1: Training all LoRA adapters"
echo "=========================================="


bash "$SCRIPT_DIR/gsm8k_qwen3_lora.sh"
bash "$SCRIPT_DIR/nlu_qwen3_lora.sh"
bash "$SCRIPT_DIR/codealpaca_qwen3_lora.sh"
bash "$SCRIPT_DIR/saferpaca_qwen3_lora.sh"
bash "$SCRIPT_DIR/mmlu_qwen3_r_32.sh"


echo ""
echo "=========================================="
echo "Step 2: Merging and evaluating adapters"
echo "=========================================="

# 切换到 lora_merge 目录并运行合并评估脚本
cd /root/autodl-tmp/lora_merge



# 运行合并和评估脚本
bash scripts/merge_and_evaluate_three_methods_qwen.sh

echo ""
echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="

# Shutdown the machine
/usr/bin/shutdown -h now