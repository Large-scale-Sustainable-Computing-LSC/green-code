import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.llama.modeling_llama_single_head_fixed_exit import LlamaEESingleHead


class TokenAccuracyPerLayer:
    def __init__(self, model_clazz, model_id, dataset_id, dataset_name, mode, max_ctx_size=350):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        if mode is not None:
            self.model = model_clazz.from_pretrained(model_id, mode=mode)
        else:
            self.model = model_clazz.from_pretrained(model_id)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.max_ctx_size = max_ctx_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        if dataset_name is not None:
            self.dataset = load_dataset(dataset_id, dataset_name)["test"]

        self.prepare_dataset()

    def calculate_perplexity(self, logits, targets):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        target_probs = probabilities.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log(target_probs)
        avg_log_prob = torch.mean(log_probs)
        perplexity = torch.exp(-avg_log_prob)
        return perplexity.item()

    def prepare_dataset(self):
        def dataset_array_to_string(example):
            example["code"] = ' '.join(example["code"][1:-1])
            return example

        self.dataset = self.dataset.map(dataset_array_to_string)

    def evaluate(self, max_new=25, num_samples=2, context_fraction=0.5):
        layer_accuracies = []
        max_context_length = self.max_ctx_size
        max_samples = num_samples
        current_samples = 0

        for snippet in tqdm(self.dataset, desc="Processing Samples"):
            if current_samples > max_samples:
                break
            current_samples += 1

            tokens = self.tokenizer(snippet["code"], return_tensors='pt').input_ids.to(self.device)
            sample_size = tokens.size(1)
            context_length = int(sample_size * context_fraction)

            if context_length > max_context_length:
                context_length = self.max_ctx_size - 1 - max_new

            for i in range(context_length, context_length + max_new):
                start_index = max(0, i - context_length)
                context = tokens[:, start_index:i].to(self.device)

                try:
                    next_token = tokens[:, i].unsqueeze(0).to(self.device)
                except IndexError:
                    break
                if len(context) == 0 or torch.Size([1, 0]) == context.shape:
                    continue
                outputs = self.model(context)
                logits_per_layer = outputs

                for layer_idx, logits in enumerate(logits_per_layer):
                    if len(layer_accuracies) <= layer_idx:
                        layer_accuracies.append([])

                    predictions = logits[:, -1, :].argmax(dim=-1)
                    accuracy = (predictions == next_token.cpu()).float().mean().item()
                    layer_accuracies[layer_idx].append(accuracy)

        avg_accuracies = [np.mean(acc) for acc in layer_accuracies]
        stds = [np.std(acc) for acc in layer_accuracies]
        return avg_accuracies, stds

    def eval_alignment_with_last_layer(self, max_new=25, num_samples=2, context_fraction=0.5):
        layer_accuracies = []
        max_context_length = self.max_ctx_size
        max_samples = num_samples
        current_samples = 0

        for snippet in tqdm(self.dataset, desc="Processing Samples"):
            if current_samples > max_samples:
                break
            current_samples += 1

            tokens = self.tokenizer(snippet["code"], return_tensors='pt').input_ids.to(self.device)
            sample_size = tokens.size(1)
            context_length = int(sample_size * context_fraction)

            if context_length > max_context_length:
                context_length = self.max_ctx_size - 1 - max_new

            for i in range(context_length, context_length + max_new):
                start_index = max(0, i - context_length)
                context = tokens[:, start_index:i].to(self.device)

                try:
                    next_token = tokens[:, i].unsqueeze(0).to(self.device)
                except IndexError:
                    continue
                if len(context) == 0 or torch.Size([1, 0]) == context.shape:
                    continue
                outputs = self.model(context)
                logits_per_layer = outputs

                last_layer = logits_per_layer[-1][:, -1, :].argmax(dim=-1)

                for layer_idx, logits in enumerate(logits_per_layer):
                    if len(layer_accuracies) <= layer_idx:
                        layer_accuracies.append([])

                    predictions = logits[:, -1, :].argmax(dim=-1)
                    accuracy = (predictions == last_layer).float().mean().item()
                    layer_accuracies[layer_idx].append(accuracy)

        avg_accuracies = [np.mean(acc) for acc in layer_accuracies]
        sd = [np.std(acc) for acc in layer_accuracies]

        return avg_accuracies, sd

    def plot_accuracies(self, avg_accuracies, filename='accuracies_plot2.png',
                        plt_title="Token-Level Accuracy Per Layer with >= 50% context size as input (first 300 test samples)"):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(avg_accuracies)), avg_accuracies, marker='o', linestyle='-')
        plt.xlabel('Layer Number')
        plt.ylabel('Accuracy')
        plt.title(plt_title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_accuracies_2(self, avg_accuracies1, avg_accuracies2, filename='accuracies_opt_125m_java_comp.png',
                          plt_title="Token-Level Accuracy Per Layer with >= 50% context size as input (first 300 test samples)"):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(avg_accuracies1)), avg_accuracies1, marker='o', linestyle='-', label='Base Model')
        plt.plot(range(len(avg_accuracies2)), avg_accuracies2, marker='s', linestyle='--', label='With weighted loss')
        plt.xlabel('Layer Number')
        plt.ylabel('Accuracy')
        plt.title(plt_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_alignment_2(self, avg_accuracies1, avg_accuracies2, std_dev1, std_dev2,
                         filename='accuracies_opt_125m_java_comp.png',
                         plt_title="Token-Level Accuracy Per Layer with >= 50% context size as input (first 300 test samples)"):
        plt.figure(figsize=(12, 6))

        # Plot accuracies
        plt.plot(range(len(avg_accuracies1)), avg_accuracies1, marker='o', linestyle='-', label='Base Model')
        plt.plot(range(len(avg_accuracies2)), avg_accuracies2, marker='s', linestyle='--', label='Finetuned')

        # Fill area for standard deviation
        plt.fill_between(range(len(avg_accuracies1)),
                         np.array(avg_accuracies1) - np.array(std_dev1),
                         np.array(avg_accuracies1) + np.array(std_dev1),
                         color='blue', alpha=0.1)  # Adjust color and alpha as needed

        plt.fill_between(range(len(avg_accuracies2)),
                         np.array(avg_accuracies2) - np.array(std_dev2),
                         np.array(avg_accuracies2) + np.array(std_dev2),
                         color='orange', alpha=0.1)  # Adjust color and alpha as needed

        plt.xlabel('Layer Number')
        plt.ylabel('Accuracy')
        plt.title(plt_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_accuracies_3(self, avg_accuracies1, avg_accuracies2, avg_accuracies3,
                          filename='accuracies_opt_125m_java_comp.png',
                          plt_title="Token-Level Accuracy Per Layer with >= 50% context size as input (first 300 test samples)"):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(avg_accuracies1)), avg_accuracies1, marker='o', color='blue', label='Without weighted loss')
        plt.plot(range(len(avg_accuracies2)), avg_accuracies2, marker='s', color='red', label='With weighted loss')
        plt.plot(range(len(avg_accuracies2)), avg_accuracies3, marker='s', color='green', label='Not finetuned')

        plt.xlabel('Layer Number')
        plt.ylabel('Accuracy')
        plt.title(plt_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def save_accuracies_to_json(self, file_path, accuracies):
        with open(file_path, 'w') as f:
            json.dump(accuracies, f)

    def load_accuracies_from_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)


def eval_llama():
    global num_samples
    evaluator = TokenAccuracyPerLayer(model_clazz=LlamaEESingleHead,
                                      model_id="meta-llama/Llama-3.2-3B",

                                      mode="return_layer_logits",
                                      dataset_id="google/code_x_glue_cc_code_completion_token", dataset_name="python")
    avg_accuracies2, stds2 = evaluator.eval_alignment_with_last_layer(max_new=15, num_samples=num_samples,
                                                                      context_fraction=0.25)
    evaluator.save_accuracies_to_json('avg_align_smaples_ctx05_llama_base.json', avg_accuracies2)
    json_ = {"acc": avg_accuracies2, "stds": stds2}
    evaluator.save_accuracies_to_json('avg_align_llama_python_maxnew15_1000_smaples_ctx025_llama_with_base.json', json_)
    evaluator.model.cpu()
    del evaluator.model
    gc.collect()
    num_samples = 1000
    evaluator = TokenAccuracyPerLayer(model_clazz=LlamaEESingleHead,
                                      model_id="XXX",

                                      mode="return_layer_logits",
                                      dataset_id="google/code_x_glue_cc_code_completion_token", dataset_name="java")
    avg_accuracies, stds = evaluator.eval_alignment_with_last_layer(max_new=15, num_samples=num_samples,
                                                                    context_fraction=0.25)
    json_ = {"acc": avg_accuracies, "stds": stds}
    evaluator.save_accuracies_to_json('avg_align_llama_java_maxnew15_1000_smaples_ctx025_llama_with_weighted.json',
                                      json_)
    evaluator.model.cpu()
    del evaluator.model
    gc.collect()
    torch.cuda.empty_cache()
    evaluator = TokenAccuracyPerLayer(model_clazz=LlamaEESingleHead,
                                      model_id="meta-llama/Llama-3.2-3B",

                                      mode="return_layer_logits",
                                      dataset_id="google/code_x_glue_cc_code_completion_token", dataset_name="java")
    avg_accuracies2, stds2 = evaluator.eval_alignment_with_last_layer(max_new=15, num_samples=num_samples,
                                                                      context_fraction=0.25)
    evaluator.save_accuracies_to_json('avg_acc_smaples_ctx05_llama_base.json', avg_accuracies2)
    json_ = {"acc": avg_accuracies2, "stds": stds2}
    evaluator.save_accuracies_to_json('avg_align_llama_java_maxnew15_1000_smaples_ctx025_llama_with_base.json', json_)
