import random
from random import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.llama.modeling_llama_rl_ee import LlamaEESingleHeadRL
from src.models.opt.modeling_opt_ee_rl import OPTEESingleHeadRLHiddenState
from src.models.rl.enviornments.sb3_torch_wrapper import SB3TorchWrapper


class EvalExitAccuracy():
    def __init__(self, model_id, rl_agent, num_layers, device, dataset, debug=True, agent_mode="PPO", agent_thresh=0.90,
                 start_sample=0, allowed_exits: List[int] = None, model_max_ctx: int = 256, cache_dir=None, min_ctx=0.3,
                 max_new=15, max_samples=None):
        super(EvalExitAccuracy, self).__init__()
        random.seed(42)
        if "llama" in model_id:
            if cache_dir:
                self.model = LlamaEESingleHeadRL.from_pretrained(model_id, cache_dir=cache_dir, mode="train_rl_gen").to(
                    device)
            else:
                self.model = LlamaEESingleHeadRL.from_pretrained(model_id, mode="train_rl_gen").to(device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        elif "opt" in model_id:
            if cache_dir:

                self.model = OPTEESingleHeadRLHiddenState.from_pretrained(model_id, cache_dir=cache_dir,
                                                                          mode="train_rl_gen").to(device)
            else:
                self.model = OPTEESingleHeadRLHiddenState.from_pretrained(model_id, mode="train_rl_gen").to(device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        else:
            raise Exception("Only llama and OPT supported")
        self.allowed_exits = allowed_exits
        self.num_layers = num_layers
        self.device = device
        self.dataset = dataset
        self.agent_thresh = agent_thresh

        self.rl_model = SB3TorchWrapper(rl_agent, mode=agent_mode).to("cuda")

        if max_samples == None:
            self.max_samples = len(self.dataset)
        else:
            self.max_samples = max_samples
        self.model_max_ctx = model_max_ctx
        self.current_sample = random.randint(0, len(self.dataset) - 1)
        self.acc_dict = {key: [] for key in self.allowed_exits}
        self.count_dict = {key: [] for key in self.allowed_exits}

        self.acc_dict[num_layers - 1] = []
        self.count_dict[num_layers - 1] = []

        self.mean_layers = []
        self.max_new = max_new

    def eval(self, soft=True):

        for sample in tqdm(range(0, self.max_samples), desc="Processing samples"):
            self.min_ctx = random.uniform(0.2, 0.4)
            self.current_text = self.dataset[sample]["code"]

            inputs = self.tokenizer(self.current_text, return_tensors='pt').to(self.device)
            self.current_input_ids = inputs['input_ids']
            self.current_attention_mask = inputs['attention_mask']
            self.current_ctx = int(self.current_input_ids.size(1) * self.min_ctx)
            if self.current_ctx >= self.model_max_ctx:
                self.current_ctx = self.model_max_ctx
            if self.current_ctx == 0:
                continue

            output = self.model.generate(self.current_input_ids[:, :self.current_ctx],
                                         attention_mask=self.current_attention_mask[:, :self.current_ctx],
                                         max_new_tokens=self.max_new, do_sample=False, num_beams=1)

            self.model_states = self.model.get_states()
            self.model.reset_states()
            sample_layers = []
            for state in self.model_states:
                self.llm_states_curr_tk = state
                first_last_match, last_layer_token = self.get_optimal_exit_point()
                has_exited = -1
                curr_layer_idx = 0
                curr_layer = self.allowed_exits[curr_layer_idx]
                while has_exited == -1:
                    next_state = self.llm_states_curr_tk[curr_layer]

                    action_logits = self.rl_model(next_state['hidden_state'][0, -1, :])
                    temperature = 1.5
                    action_probabilities = F.softmax(action_logits / temperature, dim=0)

                    action = torch.argmax(action_probabilities).item()

                    if action == 1 and action_probabilities[1] > self.agent_thresh:
                        has_exited = curr_layer
                        sample_layers.append(has_exited)

                    if curr_layer == self.allowed_exits[-1]:
                        break

                    curr_layer_idx = (curr_layer_idx + 1) % len(self.allowed_exits)
                    curr_layer = self.allowed_exits[curr_layer_idx]

                if has_exited == -1:
                    has_exited = self.num_layers - 1
                if soft:
                    pred = self.llm_states_curr_tk[has_exited]["prediction"]
                    if pred == last_layer_token:
                        self.acc_dict[has_exited].append(1)
                        self.count_dict[has_exited].append(1)
                    else:
                        self.acc_dict[has_exited].append(0)

                else:

                    if has_exited == first_last_match:
                        self.acc_dict[has_exited].append(1)
                        self.count_dict[has_exited].append(1)

                    else:
                        self.acc_dict[has_exited].append(0)
                self.mean_layers.append(np.mean(sample_layers))

        for key, value in self.acc_dict.items():
            if value:
                mean_value = np.mean(value)
                std = np.std(value)
                print(
                    f"Mean of values for key {key}: {mean_value}+-{std}, number of occ {np.sum(self.count_dict[key])}")
            else:
                print(f"Key {key} has an empty list, no mean calculated.")

        print(np.mean(self.mean_layers))

    def get_optimal_exit_point(self):
        exit_index = -1
        last_layer_token = self.llm_states_curr_tk[-1]['prediction']
        first_last_match = -1
        # print("label: ", self.tokenizer.decode(label))
        for idx, state in enumerate(self.llm_states_curr_tk):
            # print(state["prediction"], idx)
            if state["prediction"] == last_layer_token and first_last_match == -1:
                first_last_match = idx
        return first_last_match, last_layer_token


if __name__ == "__main__":
    pass
