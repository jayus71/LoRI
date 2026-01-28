#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用的多适配器合并脚本

支持任意数量的适配器、数据集，以及灵活的合并参数配置。
基于 LoRI/src/merge_5_loras.py 改进。

使用示例:
    python src/merge_adapters.py \\
        --model_name mistralai/Mistral-7B-v0.1 \\
        --adapter_paths /path/to/adapter1 /path/to/adapter2 /path/to/adapter3 \\
        --adapter_names math commonsense code \\
        --datasets commonsense,gsm8k,mmlu \\
        --combination_type cat \\
        --density 0.7 \\
        --weights 1.0 1.0 1.0 \\
        --batch_size 64 \\
        --results_path results/merge_experiment
"""

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

# Cache directory
cache_dir = os.getenv("PROJECT_CACHE", "~/.cache")

# Commonsense task list
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='通用多适配器合并与评估脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型相关
    parser.add_argument('--model_name', type=str, required=True,
                        help='基础模型名称（用于查找config/model/*.yaml）')
    parser.add_argument('--adapter_paths', type=str, nargs='+', required=True,
                        help='适配器路径列表（按顺序）')
    parser.add_argument('--adapter_names', type=str, nargs='+', default=None,
                        help='适配器名称列表（可选，默认使用 adapter_0, adapter_1, ...）')
    
    # 合并参数
    parser.add_argument('--combination_type', type=str, default='cat',
                        choices=['cat', 'linear', 'svd', 'ties', 'dare_ties', 'dare_linear'],
                        help='合并方法类型')
    parser.add_argument('--density', type=float, default=0.7,
                        help='合并密度（用于某些合并方法）')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='适配器权重列表（默认均等权重）')
    
    # 评估参数
    parser.add_argument('--datasets', type=str, default='commonsense,gsm8k',
                        help='评估数据集，逗号分隔（支持 "commonsense" 关键字展开）')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评估批次大小')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='评估运行次数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                        help='打印详细信息')
    parser.add_argument('--sample', action='store_true',
                        help='使用采样生成（而非贪婪）')
    parser.add_argument('--results_path', type=str, default='results',
                        help='结果保存路径')
    
    return parser.parse_args()


def validate_args(args):
    """验证参数合法性"""
    num_adapters = len(args.adapter_paths)
    
    # 验证 adapter_names
    if args.adapter_names is None:
        # 自动生成名称
        args.adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
        print(f"自动生成适配器名称: {args.adapter_names}")
    else:
        if len(args.adapter_names) != num_adapters:
            raise ValueError(
                f"adapter_names 数量 ({len(args.adapter_names)}) "
                f"与 adapter_paths 数量 ({num_adapters}) 不匹配"
            )
    
    # 验证 weights
    if args.weights is None:
        # 均等权重
        args.weights = [1.0] * num_adapters
        print(f"使用均等权重: {args.weights}")
    else:
        if len(args.weights) != num_adapters:
            raise ValueError(
                f"weights 数量 ({len(args.weights)}) "
                f"与 adapter_paths 数量 ({num_adapters}) 不匹配"
            )
    
    # 展开 commonsense 关键字
    args.datasets = args.datasets.replace("commonsense", ','.join(commonsense_tasks))
    args.datasets = args.datasets.split(',')
    
    return args


def load_base_model(model_name):
    """加载基础模型和 tokenizer"""
    model_yaml_path = os.path.join('config', 'model', f"{model_name}.yaml")
    
    if not os.path.exists(model_yaml_path):
        raise FileNotFoundError(
            f"模型配置文件不存在: {model_yaml_path}\n"
            f"请确保 config/model/{model_name}.yaml 存在"
        )
    
    with open(model_yaml_path, 'r') as file:
        model_config = yaml.safe_load(file)
        load_path = model_config.get('name_or_path')
        policy_dtype = getattr(torch, model_config.get('policy_dtype', 'bfloat16'))
    
    print(f"\n{'='*70}")
    print(f"加载基础模型")
    print(f"{'='*70}")
    print(f"  模型路径: {load_path}")
    print(f"  数据类型: {policy_dtype}")
    
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
    
    print(f"  ✓ 基础模型加载完成")
    print(f"{'='*70}\n")
    
    return base_model, tokenizer


def load_and_merge_adapters(base_model, adapter_paths, adapter_names, weights, 
                             combination_type, density):
    """加载并合并多个适配器"""
    print(f"\n{'='*70}")
    print(f"加载和合并适配器")
    print(f"{'='*70}")
    print(f"  适配器数量: {len(adapter_paths)}")
    print(f"  合并方法: {combination_type}")
    print(f"  密度: {density}")
    print(f"  权重: {weights}")
    print(f"{'='*70}\n")
    
    # 加载第一个适配器
    print(f"[1/{len(adapter_paths)}] 加载适配器: {adapter_names[0]}")
    print(f"  路径: {adapter_paths[0]}")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_paths[0], 
        adapter_name=adapter_names[0]
    )
    print(f"  ✓ 加载完成\n")
    
    # 加载其余适配器
    for i in range(1, len(adapter_paths)):
        print(f"[{i+1}/{len(adapter_paths)}] 加载适配器: {adapter_names[i]}")
        print(f"  路径: {adapter_paths[i]}")
        model.load_adapter(adapter_paths[i], adapter_name=adapter_names[i])
        print(f"  ✓ 加载完成\n")
    
    # 合并适配器
    print(f"{'='*70}")
    print(f"合并适配器...")
    print(f"{'='*70}")
    merge_adapter_name = "merged"
    model.add_weighted_adapter(
        adapter_names, 
        weights, 
        merge_adapter_name, 
        combination_type=combination_type, 
        density=density
    )
    model.set_adapter(merge_adapter_name)
    model.merge_and_unload()
    print(f"  ✓ 合并完成\n")
    
    return model


def evaluate_model(model, tokenizer, datasets, args):
    """评估模型在多个数据集上的表现"""
    print(f"\n{'='*70}")
    print(f"开始评估")
    print(f"{'='*70}")
    print(f"  数据集: {datasets}")
    print(f"  运行次数: {args.num_runs}")
    print(f"{'='*70}\n")
    
    results = []
    
    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n--- 运行 {run_idx + 1}/{args.num_runs} ---\n")
        
        for dataset in datasets:
            print(f"评估数据集: {dataset}")
            acc = evaluate(dataset, model, tokenizer, args)
            
            result = {
                'run': run_idx,
                'dataset': dataset,
                'accuracy': acc * 100
            }
            results.append(result)
            
            print(f"  准确率: {acc * 100:.2f}%\n")
    
    return results


def save_results(results, args):
    """保存评估结果"""
    os.makedirs(os.path.join(args.results_path, "results"), exist_ok=True)
    
    output_file = f"{args.results_path}/results/merge_adapters.txt"
    
    with open(output_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"实验配置\n")
        f.write(f"{'='*70}\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"适配器路径: {args.adapter_paths}\n")
        f.write(f"适配器名称: {args.adapter_names}\n")
        f.write(f"合并方法: {args.combination_type}\n")
        f.write(f"密度: {args.density}\n")
        f.write(f"权重: {args.weights}\n")
        f.write(f"{'='*70}\n\n")
        
        for result in results:
            f.write(
                f"运行 {result['run']}, "
                f"数据集: {result['dataset']}, "
                f"准确率: {result['accuracy']:.2f}%\n"
            )
        
        f.write(f"\n")
    
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(0)
    
    # 解析参数
    args = parse_args()
    args = validate_args(args)
    
    print(f"\n{'#'*70}")
    print(f"{'#'*70}")
    print(f"  通用多适配器合并与评估")
    print(f"{'#'*70}")
    print(f"{'#'*70}\n")
    
    # 加载基础模型
    base_model, tokenizer = load_base_model(args.model_name)
    
    # 加载并合并适配器
    model = load_and_merge_adapters(
        base_model,
        args.adapter_paths,
        args.adapter_names,
        args.weights,
        args.combination_type,
        args.density
    )
    
    # 评估模型
    results = evaluate_model(model, tokenizer, args.datasets, args)
    
    # 保存结果
    save_results(results, args)
    
    print(f"\n{'#'*70}")
    print(f"  所有任务完成！")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
