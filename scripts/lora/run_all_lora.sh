#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/nlu_llama3_lora.sh"
bash "$SCRIPT_DIR/codealpaca_llama3_lora.sh"
bash "$SCRIPT_DIR/saferpaca_llama3_lora.sh"
bash "$SCRIPT_DIR/mmlu_llama3_r_32.sh"
bash "$SCRIPT_DIR/gsm8k_llama3_lora.sh"

