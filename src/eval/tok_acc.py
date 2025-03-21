import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class TokenLevelAccEvaluator:
    def __init__(self, model_clazz, model_id, dataset_id, dataset_name, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if mode is not None:
            self.model = model_clazz.from_pretrained(model_id, mode=mode)
        else:
            self.model = model_clazz.from_pretrained(model_id)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
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

    def evaluate(self):
        perplexities = []
        accuracies = []
        max_context_length = 1024
        max_samples = 1000
        current_samples = 0
        for snippet in tqdm(self.dataset, desc="Processing Samples"):
            if current_samples > max_samples:
                break
            current_samples = current_samples + 1
            try:
                tokens = self.tokenizer.encode(snippet["code"], return_tensors='pt').to(self.device)
                sample_size = tokens.size(1)
                context_length = sample_size // 2

                if context_length > max_context_length:
                    continue

                for i in range(context_length, sample_size):
                    start_index = max(0, i - context_length)
                    context = tokens[:, start_index:i].to(self.device)
                    next_token = tokens[:, i].unsqueeze(0).to(self.device)

                    outputs = self.model(context)
                    predictions = outputs.logits[:, -1, :].argmax(dim=-1)

                    perplexity = self.calculate_perplexity(outputs.logits[:, -1, :], next_token)
                    perplexities.append(perplexity)

                    accuracy = (predictions == next_token).float().mean().item()
                    accuracies.append(accuracy)
            except Exception as e:
                print(f"Error processing snippet: {snippet}")
                print(e)
                continue

        avg_perplexity = np.mean(perplexities)
        avg_accuracy = np.mean(accuracies)

        return avg_perplexity, avg_accuracy
