#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 LoRI/bigcode/main.py 修改
用于评估合并后的 LoRA 权重在代码生成任务上的表现
"""
import os
import sys
import fnmatch
import json
import warnings
import csv
from datetime import datetime

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm

# 从同级目录导入 bigcode_eval 模块
from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False
        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path (used with --merged_weights)",
    )
    parser.add_argument(
        "--merged_weights",
        type=str,
        default=None,
        help="Path to merged weights .pt file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="unknown",
        help="Merge method name (for CSV output)",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="outputs/all_code_results.csv",
        help="CSV file path for results",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Weight scaling factor for applying merged weights",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    parser.add_argument('--sparsity_ratio', type=float, default=0.0)
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def load_merged_weights(weights_path: str):
    """加载合并后的权重文件"""
    print(f"正在加载合并权重: {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    data = torch.load(weights_path, map_location='cpu')

    if isinstance(data, dict):
        if 'merged_weights' in data:
            merged_weights = data['merged_weights']
            config = data.get('config', {})
            print(f"  权重配置: {config}")
        else:
            merged_weights = data
    else:
        raise ValueError(f"不支持的权重格式: {type(data)}")

    print(f"  加载了 {len(merged_weights)} 个层/模块的权重")

    for key, weight in list(merged_weights.items())[:3]:
        print(f"    {key}: shape={weight.shape}, dtype={weight.dtype}")
    if len(merged_weights) > 3:
        print(f"    ... 还有 {len(merged_weights) - 3} 个")

    return merged_weights


def apply_merged_weights(model, merged_weights, scale: float = 1.0):
    """将合并后的权重应用到基础模型"""
    print(f"\n正在应用合并权重到模型...")
    print(f"  合并权重数量: {len(merged_weights)}")
    print(f"  缩放因子: {scale}")

    applied_count = 0

    with torch.no_grad():
        for key, delta_W in tqdm(merged_weights.items(), desc="应用权重"):
            parts = key.split('.')
            if len(parts) != 3 or parts[0] != 'layers':
                print(f"  跳过无效 key: {key}")
                continue

            layer_idx = int(parts[1])
            module_name = parts[2]

            # 构建完整的参数路径
            if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                param_path = f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
            elif module_name in ['up_proj', 'down_proj', 'gate_proj']:
                param_path = f"model.layers.{layer_idx}.mlp.{module_name}.weight"
            else:
                print(f"  未知模块类型: {module_name}")
                continue

            try:
                param = None
                for name, p in model.named_parameters():
                    if name == param_path:
                        param = p
                        break

                if param is None:
                    print(f"  找不到参数: {param_path}")
                    continue

                delta_W = delta_W.to(param.device).to(param.dtype)

                if delta_W.shape != param.shape:
                    print(f"  形状不匹配 {key}: delta_W {delta_W.shape} vs param {param.shape}")
                    continue

                param.data.add_(delta_W, alpha=scale)
                applied_count += 1

            except Exception as e:
                print(f"  应用 {key} 失败: {e}")
                continue

    print(f"\n成功应用 {applied_count}/{len(merged_weights)} 个权重")
    return model


def save_to_csv(csv_path: str, method: str, task: str, metric_name: str,
                metric_value: float, scale: float, merged_weights_path: str):
    """保存结果到CSV文件"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': method,
        'task': task,
        'metric_name': metric_name,
        'metric_value': f"{metric_value:.4f}",
        'scale': scale,
        'merged_weights': merged_weights_path
    }

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'method', 'task', 'metric_name', 'metric_value', 'scale', 'merged_weights']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        # 代码生成和评估模式
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )

        # 确定模型路径
        if args.merged_weights and args.base_model:
            # 使用合并权重模式
            model_path = args.base_model
            print(f"使用合并权重模式: {model_path}")
        elif args.model:
            # 使用标准模型路径
            import yaml
            # 因为脚本在 bigcode/ 目录下，所以需要 ../ 访问上级目录的 config
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_yaml_path = os.path.join(script_dir, '..', 'config', 'model', f"{args.model}.yaml")
            with open(model_yaml_path, 'r') as file:
                model_config = yaml.safe_load(file)
                model_path = model_config.get('name_or_path')
        else:
            raise ValueError("Must provide either --base_model with --merged_weights, or --model")

        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "token": args.use_auth_token,
        }

        if args.load_in_8bit:
            print("Loading model in 8bit")
            model_kwargs["load_in_8bit"] = args.load_in_8bit
            model_kwargs["device_map"] = {"": accelerator.process_index}
        elif args.load_in_4bit:
            print("Loading model in 4bit")
            model_kwargs["load_in_4bit"] = args.load_in_4bit
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": accelerator.process_index}
        else:
            print(f"Loading model in {args.precision}")
            model_kwargs["torch_dtype"] = dict_precisions[args.precision]

            if args.max_memory_per_gpu:
                if args.max_memory_per_gpu != "auto":
                    model_kwargs["max_memory"] = get_gpus_max_memory(
                        args.max_memory_per_gpu, accelerator.num_processes
                    )
                    model_kwargs["offload_folder"] = "offload"
                else:
                    model_kwargs["device_map"] = "auto"
                    print("Loading model in auto mode")
            else:
                # 默认使用 auto device_map 以确保模型正确分配到GPU
                model_kwargs["device_map"] = "auto"
                print("Using auto device_map")

        if args.modeltype == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
        elif args.modeltype == "seq2seq":
            warnings.warn(
                "Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models."
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"Non valid modeltype {args.modeltype}, choose from: causal, seq2seq"
            )

        if args.left_padding:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                token=args.use_auth_token,
                padding_side="left",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                token=args.use_auth_token,
                truncation_side="left",
                padding_side="right",
            )

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))

        # 应用合并的权重或加载PEFT模型
        if args.merged_weights:
            print(f"\n加载并应用合并权重: {args.merged_weights}")
            merged_weights = load_merged_weights(args.merged_weights)
            model = apply_merged_weights(model, merged_weights, scale=args.scale)
            print("合并权重应用完成")
        elif args.peft_model:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.peft_model)
            print("Loaded PEFT model. Merging...")
            model.merge_and_unload()
            print("Merge complete.")

        evaluator = Evaluator(accelerator, model, tokenizer, args)

        if (
            args.load_generations_intermediate_paths
            and len(args.load_generations_intermediate_paths) != len(task_names)
        ):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, \
                must pass equal number of files as number of tasks"
            )

        for idx, task in enumerate(task_names):
            intermediate_generations = None
            if args.load_generations_intermediate_paths:
                with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                    intermediate_generations = json.load(f_in)

            if args.generation_only:
                if accelerator.is_main_process:
                    print("generation mode only")
                generations, references = evaluator.generate_text(
                    task, intermediate_generations=intermediate_generations
                )
                if accelerator.is_main_process:
                    save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                    save_references_path = f"references_{task}.json"
                    evaluator.save_json_files(
                        generations,
                        references,
                        save_generations_path,
                        save_references_path,
                    )
            else:
                results[task] = evaluator.evaluate(
                    task, intermediate_generations=intermediate_generations
                )

    # 保存结果
    results["config"] = vars(args)
    if not args.generation_only:
        if accelerator.is_main_process:
            if args.merged_weights:
                print(f"Model: Merged weights from {args.merged_weights}")
            else:
                print(f"Model: {results['config'].get('peft_model', args.model)}")

            for key, value in results.items():
                if key != "config":
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")

            # 保存到文本文件
            os.makedirs(os.path.join(args.metric_output_path, "results"), exist_ok=True)
            with open(f"{args.metric_output_path}/results/sr_{args.sparsity_ratio}.txt", "a") as f:
                if args.merged_weights:
                    f.write(f"Model: Merged weights - {args.method}\n")
                    f.write(f"Weights path: {args.merged_weights}\n")
                else:
                    f.write(f"Model: {results['config'].get('peft_model', args.model)}\n")

                for key, value in results.items():
                    if key != "config":
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")

            # 保存到CSV文件
            if args.merged_weights and args.csv_output:
                for task_name, task_results in results.items():
                    if task_name != "config":
                        for metric_name, metric_value in task_results.items():
                            if isinstance(metric_value, (int, float)):
                                save_to_csv(
                                    csv_path=args.csv_output,
                                    method=args.method,
                                    task=task_name,
                                    metric_name=metric_name,
                                    metric_value=metric_value,
                                    scale=args.scale,
                                    merged_weights_path=args.merged_weights
                                )
                print(f"\n结果已保存到CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
