import time
from dataclasses import dataclass

import pynvml
import torch
from stable_baselines3 import PPO
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from transformers import Pipeline, pipeline

from src.models.llama.modeling_llama_rl_ee import LlamaEESingleHeadRL

"""
Adapted version of: https://github.com/LucienShui/huggingface-vscode-endpoint-server
"""


@dataclass
class GeneratorResult:
    generated_text: str
    energy_consumed: float
    time_taken: float
    exited_layers: list


class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str | GeneratorResult:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str | GeneratorResult:
        return self.generate(query, parameters)

    def update_threshold(self, threshold):
        raise NotImplementedError


class Llama(GeneratorBase):
    def __init__(self, pretrained: str, device: str = None, device_map: str = None):
        self.pretrained: str = pretrained
        self.pipe: Pipeline = pipeline(
            "text-generation", model=pretrained, torch_dtype=torch.bfloat16, device=device, device_map=device_map)
        self.generation_config = GenerationConfig.from_pretrained(pretrained)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text


class Llama32(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True,
                                                                           cache_dir="D:\\cache")
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True,
                                                                            cache_dir="D:\\cache")
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, query: str, parameters: dict = None) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        params = {**self.default_parameter, **(parameters or {})}
        if 'return_full_text' in params:
            params.pop('return_full_text')
        if 'RL_agent_thresh' in params:
            params.pop('RL_agent_thresh')

        if 'num_predictions' in params:
            params.pop('num_predictions')
        output_ids: torch.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text


class Llama32WithAgent(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.curr_threshold = 0.5
        self.pretrained: str = pretrained
        self.device: str = device
        # self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True, cache_dir="D:\\cache")
        self.model = LlamaEESingleHeadRL.from_pretrained(pretrained, exit_indices=[7, 15, 19, 23],
                                                         threshold=self.curr_threshold,
                                                         mode="infer_ee")

        rl_model = PPO.load(
            "path_to_ppo",
            device="cuda")
        self.model.model.set_RL_model(rl_model)

        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True,
                                                                            cache_dir="D:\\cache")
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU

    def generate(self, query: str, parameters: dict = None) -> GeneratorResult:
        start_time = time.time()
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)

        params = {**self.default_parameter, **(parameters or {})}
        if 'return_full_text' in params:
            params.pop('return_full_text')
        if 'RL_agent_thresh' in params:
            params.pop('RL_agent_thresh')

        if 'num_predict' in params:
            params['max_new_tokens'] = params.get('num_predict')
            params.pop('num_predict')

        output_ids: torch.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        consumed_energy = \
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - start_energy

        generated_text = self.post_process_output(output_text[len(query):])

        layers = self.model.get_exit_layers()
        self.model.clear_exit_layers()
        return GeneratorResult(generated_text, consumed_energy, elapsed_time, layers)

    def preprocess_data(self, text):
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        return text

    def post_process_output(self, text):
        # Add newline after each ';' and '}'
        processed_text = text.replace(';', ';\r\n').replace('}', '}\r\n')
        return processed_text

    def update_threshold(self, threshold):
        if self.curr_threshold != threshold:
            self.curr_threshold = self.model.update_threshold(threshold)
            return True

        return False
