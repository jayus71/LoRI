import re
import torch
import os
from get_datasets import get_batch_iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import pad_to_length, all_gather_if_needed
from rouge_score import rouge_scorer
from peft import PeftModel
import numpy as np
import argparse
import yaml
from eval_model import evaluate
cache_dir = os.getenv("PROJECT_CACHE", "~/.cache")
commonsense_tasks = [
    'boolq',
    'piqa',
    'social_i_qa',
    'arc-challenge',
    'arc-easy',
    'openbookqa',
    'hellaswag',
    'winogrande',
]
torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--adapter_path', type=str, default=['mistralai/Mistral-7B-v0.1'], nargs='+',
                    help='Paths to adapter models')
parser.add_argument('--adapter_names', type=str, nargs='+', default=None,
                    help='Names for each adapter (optional, auto-generated if not provided)')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--sample', action='store_true')
parser.add_argument('--datasets', type=str, default='commonsense,gsm8k')
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--combination_type', type=str, default='cat',
                    help='Combination type for merging adapters')
parser.add_argument('--density', type=float, default=0.7,
                    help='Density parameter for certain merge methods')
parser.add_argument('--weights', type=float, nargs='+', default=None,
                    help='Weights for each adapter (auto-generated equal weights if not provided)')
args = parser.parse_args()

# Validate and auto-generate parameters
num_adapters = len(args.adapter_path)

# Auto-generate adapter names if not provided
if args.adapter_names is None:
    args.adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    print(f"Auto-generated adapter names: {args.adapter_names}")
else:
    if len(args.adapter_names) != num_adapters:
        raise ValueError(
            f"Number of adapter_names ({len(args.adapter_names)}) "
            f"must match number of adapter_path ({num_adapters})"
        )

# Auto-generate equal weights if not provided
if args.weights is None:
    args.weights = [1.0] * num_adapters
    print(f"Using equal weights: {args.weights}")
else:
    if len(args.weights) != num_adapters:
        raise ValueError(
            f"Number of weights ({len(args.weights)}) "
            f"must match number of adapter_path ({num_adapters})"
        )

args.datasets = args.datasets.replace("commonsense", ','.join(commonsense_tasks))
args.datasets = args.datasets.split(',')

model_yaml_path = os.path.join('config', 'model', f"{args.model_name}.yaml")
with open(model_yaml_path, 'r') as file:
    model_config = yaml.safe_load(file)
    load_path = model_config.get('name_or_path')
    policy_dtype = getattr(torch, model_config.get('policy_dtype', 'bfloat16'))

base_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=policy_dtype,
        device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(load_path)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer))
            
print(f"\n{'='*70}")
print(f"Loading and merging {num_adapters} adapters")
print(f"{'='*70}")
print(f"Adapter paths: {args.adapter_path}")
print(f"Adapter names: {args.adapter_names}")
print(f"Weights: {args.weights}")
print(f"Combination type: {args.combination_type}")
print(f"Density: {args.density}")
print(f"{'='*70}\n")

# Load first adapter
print(f"[1/{num_adapters}] Loading adapter: {args.adapter_names[0]}")
model = PeftModel.from_pretrained(
    base_model, 
    args.adapter_path[0], 
    adapter_name=args.adapter_names[0]
)

# Load remaining adapters
for i in range(1, num_adapters):
    print(f"[{i+1}/{num_adapters}] Loading adapter: {args.adapter_names[i]}")
    model.load_adapter(args.adapter_path[i], adapter_name=args.adapter_names[i])

# Merge adapters
print(f"\nMerging {num_adapters} adapters...")
merge_adapter_name = "merged"
model.add_weighted_adapter(
    args.adapter_names, 
    args.weights, 
    merge_adapter_name, 
    combination_type=args.combination_type, 
    density=args.density
)
model.set_adapter(merge_adapter_name)
model.merge_and_unload()

# 准备 CSV 输出
import csv
method_name = args.combination_type
results_dir = os.path.join(args.results_path, method_name)
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, "results.csv")

# 检查 CSV 文件是否存在，如果不存在则写入表头
file_exists = os.path.isfile(csv_file)

for run_idx in range(args.num_runs):
    for dataset in args.datasets:
        acc = evaluate(dataset, model, tokenizer, args)
        
        # 写入 CSV
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow([
                    "run", "dataset", "accuracy", 
                    "combination_type", "density", "weights", 
                    "num_adapters", "adapter_names"
                ])
                file_exists = True
            
            # 写入数据行
            writer.writerow([
                run_idx,
                dataset,
                f"{acc * 100:.2f}",
                args.combination_type,
                args.density,
                " ".join(map(str, args.weights)),
                num_adapters,
                " ".join(args.adapter_names)
            ])
        
        print(f"Run {run_idx}, Dataset: {dataset}, Accuracy: {acc * 100:.2f}%")

print(f"\n结果已保存到: {csv_file}")