import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from src.models.modeling_opt_ee_ppo import OPTEESingleHeadRL
from transformers import AutoTokenizer


class DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            print("Dataset iteration finished...")
            self.index = 0
        item = self.dataset[self.index]
        print("index ", self.index)
        self.index += 1
        return item


class OptimalExitPointEval:
    def __init__(self, model_id, num_layers, device, dataset, debug=True, num_samples=100):

        self.model = OPTEESingleHeadRL.from_pretrained(model_id, mode="train_rl").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.iter = DatasetIterator(dataset)
        self.device = device
        self.num_layers = num_layers
        self.num_samples = num_samples

        self.DEBUG = debug

    def get_opt_exit_points(self, max_context_length=512, max_new_tokens=15):
        opt_exit_points = []
        first_last_matches = []

        for i, sample in enumerate(self.iter):
            if self.num_samples > 0 and i >= self.num_samples:
                break

            tokens = self.tokenizer.encode(sample["code"], return_tensors='pt').to(self.device)
            sample_size = tokens.size(1)
            context_length = sample_size // 2

            if context_length > max_context_length:
                continue

            sample_size = context_length + max_new_tokens
            for i in range(context_length, sample_size):
                start_index = max(0, i - context_length)
                context = tokens[:, start_index:i].to(self.device)

                next_token = tokens[:, i].unsqueeze(0).to(self.device)

                outputs = self.model(context)

                opt_exit_index, first_last_match, last_layer_token = self.get_optimal_exit_point(next_token, outputs)
                first_last_matches.append(first_last_match)
                opt_exit_points.append(opt_exit_index)

        self.plot(first_last_matches, opt_exit_points)

        return opt_exit_points

    def plot(self, first_last_matches, opt_exit_points):
        occurrences_exit_points = [0] * (self.num_layers + 2)  # +2 for -1 (no hit) and an additional count
        for point in opt_exit_points:
            if point == -1:
                occurrences_exit_points[0] += 1
            else:
                occurrences_exit_points[point + 1] += 1
        occurrences_first_last_matches = [0] * (self.num_layers + 2)
        for idx, match in enumerate(first_last_matches):
            if match == 1:
                if opt_exit_points[idx] == -1:
                    occurrences_first_last_matches[0] += 1
                else:
                    occurrences_first_last_matches[opt_exit_points[idx] + 1] += 1
        labels = ['No Hit'] + [f'Layer {i}' for i in range(self.num_layers + 1)]
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(labels))
        plt.bar(index, occurrences_exit_points, bar_width, label='Opt Exit Points', color='skyblue')
        plt.bar([i + bar_width for i in index], occurrences_first_last_matches, bar_width, label='First Last Matches',
                color='orange')
        plt.xlabel('Exit Points')
        plt.ylabel('Occurrences')
        plt.title('Occurrences of Optimal Exit Points and First Last Matches')
        plt.xticks([i + bar_width / 2 for i in index], labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Occurrences of Optimal Exit Points: ", occurrences_exit_points)
        print("Occurrences of First Last Matches: ", occurrences_first_last_matches)

        print("Relative Occurrences of Optimal Exit Points: ",
              [occ / len(opt_exit_points) for occ in occurrences_exit_points])
        print("Relative Occurrences of First Last Matches: ",
              [occ / len(first_last_matches) for occ in occurrences_first_last_matches])

    def get_optimal_exit_point2(self, label, logits):
        opt_exit_index = - 1
        last_layer_token = logits[-1][:, -1, :].argmax(dim=-1)
        first_last_match = -1
        for idx, state in enumerate(logits):
            pred = state[:, -1, :].argmax(dim=-1)
            if pred == label and opt_exit_index == - 1:
                opt_exit_index = idx

            if pred == last_layer_token and first_last_match == - 1:
                first_last_match = idx
        return opt_exit_index, first_last_match, last_layer_token

    def get_optimal_exit_point(self, label, states):
        opt_exit_index = - 1
        last_layer_token = states[-1]['prediction']
        first_last_match = -1
        for idx, state in enumerate(states):
            if state["prediction"] == label and opt_exit_index == - 1:
                opt_exit_index = idx

            if state["prediction"] == last_layer_token and first_last_match == - 1:
                first_last_match = idx
        return opt_exit_index, first_last_match, last_layer_token
