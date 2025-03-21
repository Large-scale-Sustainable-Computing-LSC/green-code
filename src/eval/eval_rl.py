import numpy as np
import torch

from src.models.opt.modeling_opt_ee_rl import OPTEESingleHeadRLHiddenState


class RLEvaluator:

    def __init__(self, model, baseline_model, tokenizer, dataset):
        self.model = model.to('cuda')
        self.base_model = baseline_model.to('cuda')
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.prepare_dataset()
        self.warmup_model()

    def prepare_dataset(self):
        def dataset_array_to_string(example):
            example["code"] = ' '.join(example["code"][1:-1])
            return example

        self.dataset = self.dataset.map(dataset_array_to_string)

    def warmup_model(self):
        for i in range(0, 10):
            with torch.no_grad():
                code_tokens = self.tokenizer.encode("System.out.println(")
                output = self.model.generate(input_ids=torch.tensor(code_tokens).unsqueeze(0).to('cuda')
                                             , max_length=15, do_sample=False)

    def get_alignment_with_baseline(self, max_ctx_size=1024, context_fraction=0.5, new_tokens=15, max_samples=100):
        matching = 0
        total_tks = 0
        if max_samples == None:
            max_samples = len(self.dataset)
        for i in range(0, max_samples):
            code = self.dataset[i]['code']
            code_tokens = self.tokenizer.encode(code)

            if len(code_tokens) > max_ctx_size:
                continue
            code_tokens = torch.tensor(code_tokens).to('cuda')
            context_size = int(len(code_tokens) * context_fraction)
            context_tokens = code_tokens[:context_size]
            input_ids = context_tokens.clone().detach().unsqueeze(0).to('cuda')
            with torch.no_grad():
                output = self.model.generate(input_ids=input_ids, max_length=context_size + new_tokens, do_sample=False)
                baseline_output = self.base_model.generate(input_ids=input_ids, max_length=context_size + new_tokens,
                                                           do_sample=False)
                new_generated_tokens = output[0][len(context_tokens):len(context_tokens) + new_tokens]
                new_baseline_tokens = baseline_output[0][len(context_tokens):len(context_tokens) + new_tokens]
                total_tks += new_tokens
                for j in range(0, new_tokens):
                    if new_generated_tokens[j] == new_baseline_tokens[j]:
                        matching += 1
        return matching / total_tks

    def count_layers_used(self, model_id, max_ctx_size=1024, context_fraction=0.5, new_tokens=15, max_samples=100):
        if type(self.model) != OPTEESingleHeadRLHiddenState:
            raise ValueError("Model must be OPTEESingleHeadRLHiddenState or TODO")

        list_of_layers = []
        if max_samples == None:
            max_samples = len(self.dataset)
        for i in range(0, max_samples):
            code = self.dataset[i]['code']
            code_tokens = self.tokenizer.encode(code)

            if len(code_tokens) > max_ctx_size:
                continue
            code_tokens = torch.tensor(code_tokens).to('cuda')
            context_size = int(len(code_tokens) * context_fraction)
            context_tokens = code_tokens[:context_size]
            input_ids = context_tokens.clone().detach().unsqueeze(0).to('cuda')
            with torch.no_grad():
                for i in range(0, new_tokens):
                    _ = self.model(input_ids=input_ids, max_length=context_size + new_tokens, do_sample=False)
                    list_of_layers.append(self.model.get_last_exit_layer_used())

        layer_counts = {}
        for layer in list_of_layers:
            if layer in layer_counts:
                layer_counts[layer] += 1
            else:
                layer_counts[layer] = 1

        return layer_counts, np.mean(list_of_layers)
