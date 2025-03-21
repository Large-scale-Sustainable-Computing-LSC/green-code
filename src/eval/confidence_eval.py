import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from src.models.modeling_opt_ee_ppo import OPTEESingleHeadRL
from transformers import AutoTokenizer


class ConfidencePlotter:

    def __init__(self, model, tokenizer, dataset):
        self.model = model.to('cuda')
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.prepare_dataset()

    def prepare_dataset(self):
        def dataset_array_to_string(example):
            example["code"] = ' '.join(example["code"][1:-1])
            return example

        self.dataset = self.dataset.map(dataset_array_to_string)

    def evaluate(self, max_context_length=50, max_new_tokens=15, max_samples=50):

        confidence_per_layer = []

        for i, snippet in enumerate(self.dataset):
            if len(confidence_per_layer) == 0:
                for _ in range(self.model.config.num_hidden_layers):
                    confidence_per_layer.append([])

            if i > max_samples:
                break

            tokens = self.tokenizer.encode(snippet["code"], return_tensors='pt').to('cuda')
            sample_size = tokens.size(1)
            context_length = sample_size // 2

            if context_length > max_context_length:
                continue

            for i in range(context_length, context_length + max_new_tokens):

                with torch.no_grad():
                    start_index = max(0, i - context_length)
                    context = tokens[:, start_index:i].to('cuda')
                    next_token = tokens[:, i].unsqueeze(0).to('cuda')
                    outputs = self.model(context)
                    logits_per_layer = outputs
                    for i, logits in enumerate(logits_per_layer):
                        # probability
                        props = F.softmax(logits, dim=-1)
                        max_probs, max_indices = torch.max(props, dim=-1)

                        confidence_per_layer[i].append(max_probs.cpu().numpy())
                    print("mean confidence per layer: ", [np.mean(layer) for layer in confidence_per_layer])

        # plot confidence per layer
        for i, layer in enumerate(confidence_per_layer):
            plt.plot(layer, label=f'Layer {i}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pass
