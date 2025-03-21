import json
import os
import random
from random import random
from typing import List, Callable

import gymnasium as gym
import numpy as np
import torch
from datasets import load_dataset
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from huggingface_hub import login
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from transformers import AutoTokenizer

from src.models.llama.modeling_llama_rl_ee import LlamaEESingleHeadRL
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


class EnvHiddenState(gym.Env):
    def __init__(self, model_id, num_layers, device, dataset, debug=True, allowed_exits: List[int] = None,
                 model_max_ctx: int = 1024, cache_dir=None, min_ctx=0.3, max_new=15, max_samples=None,
                 alpha=1.0, beta=1.0, gamma=1.0, eps=0.1):
        super(EnvHiddenState, self).__init__()
        random.seed(42)
        if "llama" in model_id:
            if cache_dir:
                print("here")
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
        self.action_dim = 2  # continue, exit

        self.num_layers = num_layers
        self.device = device
        self.dataset = dataset
        self.iter = DatasetIterator(dataset)

        if max_samples == None:
            self.max_samples = len(self.dataset)
        else:
            self.max_samples = max_samples

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        self.current_step = 0
        self.current_token = 0  # points to current label token
        self.done = False
        self.wrong_cont = 0
        self.llm_states_curr_tk = []
        self.state = None
        self.layer_state = 0
        self.last_action = 1
        self.model_max_ctx = model_max_ctx
        self.current_sample = random.randint(0, len(self.dataset) - 1)

        # Initialize states and input
        self.current_text = self.dataset[self.current_sample]["code"]
        self.min_ctx = min_ctx
        self.current_input_ids = self.tokenizer.encode(self.current_text, return_tensors='pt').to(self.device)
        self.current_ctx = int(self.current_input_ids.size(1) * min_ctx)
        if self.current_ctx > self.model_max_ctx:
            self.current_ctx = self.model_max_ctx

        self.max_new = max_new
        output = self.model.generate(self.current_input_ids[:, :self.current_ctx], max_new_tokens=max_new,
                                     min_new_tokens=self.max_new)
        self.model_states = self.model.get_states()
        self.llm_states_curr_tk = self.model_states[0]
        self.model.reset_states()
        self.states_length = len(self.model_states)
        self.current_state = 0

        self.DEBUG = debug

        self.action_space = spaces.Discrete(self.action_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.get_hidden_size(),), dtype=np.float32
        )

        self.current_energy = 0
        self.current_time = 0

        self.recent_energy = []
        self.recent_time = []

        self.last_episode_steps = 0

        self.last_exit_pos = []

        self.allowed_exits = allowed_exits
        self.layer_state = allowed_exits[0] if allowed_exits is not None else 0
        self.allowed_exits_idx = 0
        self.last_episode_reward = 0
        self.eps_exits = []
        self.opt_exits = []

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.min_ctx = random.uniform(0.2, 0.5)
        if self.last_episode_steps == 0:
            self.last_episode_steps = 1
        logger.info(
            f"Reset, ctx: {self.min_ctx}, last episode steps:{self.last_episode_steps}, reward:{self.last_episode_reward}, mean rew/step:{self.last_episode_reward / self.last_episode_steps}")
        logger.info(f"Episode exits: {self.eps_exits}")
        logger.info(f"Optimal exits: {self.opt_exits}")
        self.eps_exits = []
        self.opt_exits = []
        self.last_episode_reward = 0
        self.last_episode_steps = 0
        self.allowed_exits_idx = 0
        is_ok_sample = False
        while not is_ok_sample:

            self.current_sample = random.randint(0, len(self.dataset) - 1)
            # Reset environment state
            self.current_text = self.dataset[self.current_sample]["code"]
            inputs = self.tokenizer(self.current_text, return_tensors='pt').to(self.device)
            self.current_input_ids = inputs['input_ids']
            self.current_attention_mask = inputs['attention_mask']

            self.current_ctx = int(self.current_input_ids.size(1) * self.min_ctx)
            if self.current_ctx >= self.model_max_ctx:
                self.current_ctx = self.model_max_ctx

            if self.current_ctx > 0:
                is_ok_sample = True
            else:
                continue
            logger.info(f"New sample {self.current_sample} context size: {self.current_ctx}")
            self.current_step = 0
            self.current_token = 0
            self.done = False
            self.layer_state = self.allowed_exits[0] if self.allowed_exits is not None else 0

            output = self.model.generate(self.current_input_ids[:, :self.current_ctx],
                                         attention_mask=self.current_attention_mask[:, :self.current_ctx],
                                         max_new_tokens=self.max_new, do_sample=False, num_beams=1)
            self.model_states = self.model.get_states()
            self.llm_states_curr_tk = self.model_states[0]

            self.model.reset_states()
            self.states_length = len(self.model_states)
            next_state = self.llm_states_curr_tk[self.layer_state]

            if self.DEBUG:
                print(f"next_state shape: {next_state['hidden_state'].shape}")
                print(f"next_state: {next_state}")

            hidden_state = next_state['hidden_state'][0, -1, :].detach().cpu().numpy()

            print("last exit positions mean: ", np.mean(self.last_exit_pos))
            self.last_exit_pos = []

            return hidden_state, {}

    def step(self, action):
        if self.done:
            raise Exception("Step called on a done environment. Please reset.")

        curr_state = self.llm_states_curr_tk[self.layer_state]

        first_last_match, last_layer_token = self.get_optimal_exit_point()

        if first_last_match == -1:
            first_last_match = self.num_layers - 1

        if self.DEBUG:
            print("-" * 50)

        reward = self.compute_reward(curr_state, action, first_last_match, last_layer_token,
                                     )

        logger.info(
            f"optimal layer:  {first_last_match}  current layer: {self.layer_state}  reward:  {reward}  action:  {action}")
        if self.DEBUG:
            print(
                f"context: {self.tokenizer.decode(self.current_input_ids[0, :self.current_ctx + self.current_token + 1])}")
            print(
                f"action: {action}, curr_token: {self.current_token}, curr_step: {self.current_step}, curr_layer: {self.layer_state}")
            print(
                f"optimal_exit_layer_for_token: {first_last_match}, first_last_match: {first_last_match}, last_layer_token: {last_layer_token}")
            print(f"last-layer-tk: {self.tokenizer.decode(last_layer_token)}")
            print(f"prediction: {curr_state['prediction']}: {self.tokenizer.decode(curr_state['prediction'])}")
            print(
                f"preiction all layers: {[self.tokenizer.decode(state['prediction']) for state in self.llm_states_curr_tk]}")
            print(f"reward: {reward}")
            print("-" * 50)

        if action >= 1:
            self.current_step += 1

            if self.current_token + 1 < self.states_length:
                self.current_token += 1

                self.llm_states_curr_tk = self.model_states[self.current_token]

                self.eps_exits.append(self.layer_state)
                self.opt_exits.append(first_last_match)
                self.last_exit_pos.append(self.allowed_exits[self.allowed_exits_idx])
                # reset layer
                self.allowed_exits_idx = 0
                self.layer_state = self.allowed_exits[self.allowed_exits_idx]

            else:
                self.eps_exits.append(self.layer_state)
                self.opt_exits.append(first_last_match)
                return None, reward, True, False, {}

        else:  # continue

            if self.layer_state == self.allowed_exits[-1] and (self.current_token + 1) < self.states_length:
                self.layer_state = self.allowed_exits[0]
                self.current_token += 1
                self.current_step += 1
                self.llm_states_curr_tk = self.model_states[self.current_token]
                if first_last_match == self.num_layers - 1:
                    reward = 1
            elif self.layer_state == self.allowed_exits[-1]:  #
                if first_last_match == self.num_layers - 1:
                    reward = 1
                return None, reward, True, False, {}
            else:
                self.allowed_exits_idx = (self.allowed_exits_idx + 1) % len(self.allowed_exits)
                self.layer_state = self.allowed_exits[self.allowed_exits_idx]

        if self.current_step > 14:
            self.done = True

        next_state = self.llm_states_curr_tk[self.layer_state]

        hidden_state = next_state['hidden_state'][0, -1, :].detach().cpu().numpy()
        self.last_episode_steps += 1
        self.last_episode_reward += reward
        return hidden_state, reward, self.done, False, {}

    def is_best_exit(self, optimal_exit_layer, curr_layer):

        if optimal_exit_layer not in self.allowed_exits:
            for exit_layer in self.allowed_exits:
                if exit_layer > optimal_exit_layer:
                    return curr_layer == exit_layer
            return False
        else:
            return curr_layer == optimal_exit_layer

    def compute_reward(self, state, action, first_last_match, last_layer_pred):
        curr_layer_norm = state['state'][1]
        optimal_exit_layer_norm = first_last_match / (self.num_layers - 1)

        reward = 0
        if action == 1:
            if self.is_best_exit(first_last_match, curr_layer_norm * (self.num_layers - 1)):
                reward = 1
            elif state['prediction'] == last_layer_pred:
                reward -= (curr_layer_norm - optimal_exit_layer_norm) * (self.num_layers - 1) * self.alpha
            else:
                if curr_layer_norm < optimal_exit_layer_norm:
                    penalty = (optimal_exit_layer_norm - curr_layer_norm) * (self.num_layers - 1)
                    reward -= penalty * self.beta
                else:
                    reward -= self.eps

        else:  # Continue
            if curr_layer_norm < optimal_exit_layer_norm:
                reward = 1

            else:
                reward -= (self.get_next_layer_norm(curr_layer_norm) - optimal_exit_layer_norm) * (
                            self.num_layers - 1) * self.gamma

        reward = self.min_max_scaling_to_range(reward, -(self.allowed_exits[-1] * self.gamma), 0, -1, 0)

        return reward.item() if isinstance(reward, torch.Tensor) else reward

    def get_next_layer_norm(self, layer_norm):
        layer_num = layer_norm * (self.num_layers - 1)

        for layer in self.allowed_exits:
            if layer > layer_num:
                return layer / (self.num_layers - 1)
        return 1

    def min_max_scaling_to_range(self, data, min_val, max_val, min_range=-1, max_range=1):
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data * (max_range - min_range) + min_range

    def get_optimal_exit_point(self):
        exit_index = -1
        last_layer_token = self.llm_states_curr_tk[-1]['prediction']
        first_last_match = -1
        for idx, state in enumerate(self.llm_states_curr_tk):
            if state["prediction"] == last_layer_token and first_last_match == -1:
                first_last_match = idx
        return first_last_match, last_layer_token


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0, file_path="rewards.json"):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.file_path = file_path

    def _on_step(self) -> bool:
        # Convert rewards to Python float to avoid serialization issues
        self.rewards.append(float(self.locals['rewards'][0]))
        return True

    def _on_rollout_end(self) -> None:
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

        with open(self.file_path, 'r') as f:
            cumulative_rewards = json.load(f)

        cumulative_rewards.extend(self.rewards)

        # Save the updated rewards to the file
        with open(self.file_path, 'w') as f:
            json.dump(cumulative_rewards, f)

        # Clear the rewards list
        self.rewards.clear()


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


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
