import os
from collections import deque

import gymnasium as gym
import numpy as np
import pynvml
import torch
from datasets import load_dataset
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from huggingface_hub import login
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from transformers import AutoTokenizer

from src.models.opt.modeling_opt_ee_rl import OPTEESingleHeadRLHiddenState


class DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            print("Dataset iteration finished...")
            self.index = 0  # Reset the index to start from the beginning
        item = self.dataset[self.index]
        self.index += 1
        return item


@deprecated
class HiddenStateEnv(gym.Env):
    def __init__(self, model_id, num_layers, device, dataset, debug=True, start_sample=0):
        super(HiddenStateEnv, self).__init__()

        self.model = OPTEESingleHeadRLHiddenState.from_pretrained(model_id, cache_dir="XX", mode="train_rl").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.state_dim = 3
        self.action_dim = 2
        self.num_layers = num_layers
        self.device = device
        self.dataset = dataset
        self.iter = DatasetIterator(dataset)

        self.current_text = next(self.iter)["code"]
        self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
        self.current_step = 0
        self.current_token = 0
        self.done = False

        self.llm_states_curr_tk = []
        self.state = None
        self.current_states_idx = 0
        self.last_action = 1

        # Initialize states and input
        self.current_text = next(self.iter)["code"]
        self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
        self.llm_states_curr_tk = self.model(self.current_input_ids[:, :1])

        self.current_sample = start_sample

        self.DEBUG = debug

        # Action space: 0 (continue), 1 (exit), 2 (abort-exit)
        self.action_space = spaces.Discrete(self.action_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.get_hidden_size(),), dtype=np.float32
        )

        self.current_energy = 0
        self.current_time = 0

        self.recent_energy = []
        self.recent_time = []
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.init_recent_energy_time()
        self.max_eps_token = 1024
        self.curr_eps_token = 0

    def init_recent_energy_time(self):
        # Warm up and initialize computation graph
        for i in range(0, 100):
            _ = self.model(self.current_input_ids[:, :i + 1])

        for _ in range(5):
            with torch.no_grad():
                out = self.model(self.current_input_ids[:, :150])
                for state in out:
                    self.recent_energy.append(state['energy'])
                    self.recent_time.append(state['time'])

        print("Mean time: ", np.mean(self.recent_time))
        print("Mean energy: ", np.mean(self.recent_energy))

    def reset(self, seed=None, options=None):
        # print("Resetting environment...")
        super().reset(seed=seed)

        self.current_sample = (self.current_sample + 1) % len(self.dataset)

        # Reset environment state
        self.current_text = self.dataset[self.current_sample]["code"]
        self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
        self.current_step = 0
        self.current_token = 0
        self.curr_eps_token = 0  # reset episode counter
        self.done = False
        self.current_states_idx = 0

        next_state = self.llm_states_curr_tk[self.current_states_idx]
        if self.DEBUG:
            print(f"next_state shape: {next_state['hidden_state'].shape}")
            print(f"next_state: {next_state}")

        hidden_state = next_state['hidden_state'][0, -1, :].detach().cpu().numpy()

        return hidden_state, {}

    def step(self, action):
        if self.done:
            raise Exception("Step called on a done environment. Please reset.")

        if self.current_token > self.max_eps_token:
            print("Resetting environment due to max context length")
            return None, 0, True, False, {}

        curr_state = self.llm_states_curr_tk[self.current_states_idx]

        label = self.current_input_ids[:, self.current_token + 1]
        optimal_exit_layer_for_token, first_last_match, last_layer_token = self.get_optimal_exit_point(label)

        if self.DEBUG:
            print("-" * 50)

        reward = self.compute_reward(curr_state, action, label, first_last_match, last_layer_token,
                                     optimal_exit_layer_for_token)

        if self.DEBUG:
            print(f"context: {self.tokenizer.decode(self.current_input_ids[0, :self.current_token + 1])}")
            print(
                f"action: {action}, curr_token: {self.current_token}, curr_step: {self.current_step}, curr_layer: {self.current_states_idx}")
            print(
                f"optimal_exit_layer_for_token: {optimal_exit_layer_for_token}, first_last_match: {first_last_match}, last_layer_token: {last_layer_token}")
            print(f"last-layer-tk: {self.tokenizer.decode(last_layer_token)}")
            print(f"label: {label}: {self.tokenizer.decode(label)}")
            print(f"prediction: {curr_state['prediction']}: {self.tokenizer.decode(curr_state['prediction'])}")
            print(
                f"preiction all layers: {[self.tokenizer.decode(state['prediction']) for state in self.llm_states_curr_tk]}")
            print(f"reward: {reward}")
            print("-" * 50)

        if self.current_token >= self.current_input_ids.size(1) - 2 and self.curr_eps_token < self.max_eps_token:
            self.current_sample = (self.current_sample + 1) % len(self.dataset)

            # Reset environment state
            self.current_text = self.dataset[self.current_sample]["code"]
            self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
            self.current_step = 0
            self.current_token = 0
            self.done = False
            self.current_states_idx = 0

            next_state = self.llm_states_curr_tk[self.current_states_idx]
            if self.DEBUG:
                print(f"next_state shape: {next_state['hidden_state'].shape}")
                print(f"next_state: {next_state}")
        if action >= 1:
            self.current_token += 1
            self.curr_eps_token += 1

            with torch.no_grad():
                self.llm_states_curr_tk = self.model(self.current_input_ids[:, :self.current_token + 1])
                """   for state in self.llm_states_curr_tk:
                        self.recent_energy.append(state['energy'])
                        self.recent_time.append(state['time'])
                    if len(self.recent_energy) > 200:
                        self.recent_energy = self.recent_energy[50:]
                        self.recent_time = self.recent_time[50:]
                 """
            self.current_states_idx = 0
            if self.current_token >= self.current_input_ids.size(1) - 2 and self.curr_eps_token < self.max_eps_token:
                # Since we want a fixed number of tokens per episode (to enable fixed length [in num tks] episodes)
                self.current_sample = (self.current_sample + 1) % len(self.dataset)

                # Reset environment state for new sample processing
                self.current_text = self.dataset[self.current_sample]["code"]
                self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
                self.current_step = 0
                self.current_token = 0
                self.done = False
                self.current_states_idx = 0
                print("reset without reset")

        else:
            self.current_states_idx += 1
            if self.current_states_idx == self.num_layers:
                self.current_states_idx = 0
                self.current_token += 1
                reward -= 1  # Extra penalty for continuing after last layer
        if self.curr_eps_token >= self.max_eps_token:
            self.done = True

        next_state = self.llm_states_curr_tk[self.current_states_idx]
        self.current_step += 1

        hidden_state = next_state['hidden_state'][0, -1, :].detach().cpu().numpy()

        return hidden_state, reward, self.done, False, {}

    def compute_reward(self, state, action, label, first_last_match, last_layer_pred, optimal_exit_layer_for_token):
        curr_layer_norm = state['state'][1]

        optimal_exit_layer_norm = optimal_exit_layer_for_token / (self.num_layers - 1)
        reward = 0
        if action == 1:
            if state['prediction'] == label:
                reward += 1

                if curr_layer_norm == optimal_exit_layer_norm:
                    reward += 1
                elif curr_layer_norm > optimal_exit_layer_norm:
                    reward -= curr_layer_norm
                else:
                    reward = -1

        return reward

    def get_optimal_exit_point(self, label):
        exit_index = -1
        last_layer_token = self.llm_states_curr_tk[-1]['prediction']
        first_last_match = -1
        for idx, state in enumerate(self.llm_states_curr_tk):
            if state["prediction"] == label and exit_index == -1:
                exit_index = idx
            if state["prediction"] == last_layer_token and first_last_match == -1:
                first_last_match = idx

        return exit_index, first_last_match, last_layer_token


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = deque(maxlen=64)
        self.cumulative_rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])

        if len(self.rewards) == 64:
            cumulative_reward = sum(self.rewards)
            self.cumulative_rewards.append(cumulative_reward)

        return True


class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_step_{self.n_calls}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        return True


if __name__ == "__main__":
    pass
