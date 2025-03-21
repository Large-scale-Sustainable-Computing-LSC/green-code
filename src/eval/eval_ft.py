import gc
import json
import os

import datasets
import torch
from huggingface_hub import login
from transformers import AutoTokenizer

from src.eval.evaluator import Evaluator
from src.models.llama.modeling_llama_single_head_fixed_exit import LlamaEESingleHead
from src.models.opt.modeling_opt_single_head_fixed_exit import OPTFixedEESingleHead


def run_eval_opt(model_id, max_new, num_samples, ctx, output_file, num_layers, dataset, tokenizer):
    results = []

    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, mode="infer_ee", device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)

        codebleu_optee, chrf_optee, rouge_optee, bleu_optee, rest = evaluator.evaluate(
            max_new=max_new, num_samples=num_samples,
            context_fraction=ctx
        )

        result = {
            'exit_index': i,
            'codebleu_optee': codebleu_optee,
            'chrf_optee': chrf_optee,
            'rouge_optee': rouge_optee,
            'bleu_optee': bleu_optee,
            'rest': rest
        }

        results.append(result)
        model.cpu()
        del model
        gc.collect()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def run_eval(model_id, max_new, num_samples, ctx, output_file, num_layers, dataset, tokenizer):
    results = []

    for i in range(0, num_layers):
        model = LlamaEESingleHead.from_pretrained(model_id, mode="infer_ee", device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)

        codebleu_optee, chrf_optee, rouge_optee, bleu_optee, rest = evaluator.evaluate(
            max_new=max_new, num_samples=num_samples,
            context_fraction=ctx
        )

        result = {
            'exit_index': i,
            'codebleu_optee': codebleu_optee,
            'chrf_optee': chrf_optee,
            'rouge_optee': rouge_optee,
            'bleu_optee': bleu_optee,
            'rest': rest
        }

        results.append(result)
        model.cpu()
        del model
        gc.collect()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def run_eval_for_all_baseline(model_id, max_new, num_samples, ctx, output_file):
    results = []

    for i in range(0, 28):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        with torch.no_grad():
            model = LlamaEESingleHead.from_pretrained(model_id, mode="infer_ee", device_map="cuda:0", exit_index=i)
            model.eval()
            evaluator = Evaluator(model, tokenizer, dataset, max_ctx_size=512)

        codebleu_optee, chrf_optee, rouge_optee, bleu_optee, time_total, energy_total, mean_time_optee, mean_energy_optee = evaluator.evaluate(
            max_new=max_new, num_samples=num_samples, context_fraction=ctx
        )
        print(f"layer: {i} , codebleu:{codebleu_optee}")

        result = {
            'checkpoint': None,
            'exit_index': i,
            'codebleu_optee': codebleu_optee,
            'chrf_optee': chrf_optee,
            'rouge_optee': rouge_optee,
            'bleu_optee': bleu_optee,
            'time_total': time_total,
            'energy_total': energy_total,
            'mean_time_optee': mean_time_optee,
            'mean_energy_optee': mean_energy_optee
        }

        results.append(result)

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def run_eval_for_all_checkpoints(checkpoints_folder, max_new, num_samples, ctx, output_file):
    results = []

    checkpoint_dirs = [d for d in os.listdir(checkpoints_folder)
                       if os.path.isdir(os.path.join(checkpoints_folder, d)) and d != "runs"]

    for checkpoint_dir in checkpoint_dirs:
        model_path = os.path.join(checkpoints_folder, checkpoint_dir)
        print(f"Evaluating checkpoint: {model_path}")

        for i in range(0, 28):
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            with torch.no_grad():
                model = LlamaEESingleHead.from_pretrained(model_path, mode="infer_ee", device_map="cuda:0",
                                                          exit_index=i)
                model.eval()
                evaluator = Evaluator(model, tokenizer, dataset, max_ctx_size=512)

            codebleu_optee, chrf_optee, rouge_optee, bleu_optee, time_total, energy_total, mean_time_optee, mean_energy_optee = evaluator.evaluate(
                max_new=max_new, num_samples=num_samples, context_fraction=ctx
            )
            print(f"layer: {i} , codebleu:{codebleu_optee}")

            result = {
                'checkpoint': checkpoint_dir,
                'exit_index': i,
                'codebleu_optee': codebleu_optee,
                'chrf_optee': chrf_optee,
                'rouge_optee': rouge_optee,
                'bleu_optee': bleu_optee,
                'time_total': time_total,
                'energy_total': energy_total,
                'mean_time_optee': mean_time_optee,
                'mean_energy_optee': mean_energy_optee
            }

            results.append(result)

            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# main
