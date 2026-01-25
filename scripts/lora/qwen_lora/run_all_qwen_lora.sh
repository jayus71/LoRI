#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/nlu_qwen3_lora.sh"
bash "$SCRIPT_DIR/codealpaca_qwen3_lora.sh"
bash "$SCRIPT_DIR/saferpaca_qwen3_lora.sh"
bash "$SCRIPT_DIR/mmlu_qwen3_r_32.sh"
bash "$SCRIPT_DIR/gsm8k_qwen3_lora.sh"

