from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pynvml
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from torch import nn
from torch.cuda import temperature
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM, LlamaConfig, Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from zeus.monitor import ZeusMonitor

from src.models.rl.enviornments.sb3_torch_wrapper import SB3TorchWrapper


@dataclass
class BaseModelOutputWithPastAndExitLayer(BaseModelOutputWithPast):
    exit_layer: int = None


@dataclass
class BaseModelOutputWithPastExitAndStates(BaseModelOutputWithPastAndExitLayer):
    states: [] = None
    logits: Optional[torch.FloatTensor] = None
    has_exited: Optional[bool] = None


class LlamaModelWithExit(LlamaModel):

    def __init__(self, config: LlamaConfig, mode: str, exit_indices: List[int] = None, lm_head=None,
                 temperature: float = 1):
        super().__init__(config)
        self.exit_indices = exit_indices
        self.mode = mode
        self.lm_head = lm_head
        self.num_hidden_layers = config.num_hidden_layers

        self.rl_model = None
        self.post_init()
        self.temperature = temperature

        self.monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)

        self.rl_gpu_energy = []
        self.rl_latency = []

    def set_RL_model(self, rl_model: SelfBaseAlgorithm, mode="PPO"):
        self.rl_model = SB3TorchWrapper(rl_model, mode=mode).to("cuda")

    def get_rl_energy(self):
        return self.rl_gpu_energy

    def get_rl_energy_time(self):
        return self.rl_gpu_energy, self.rl_latency

    def clear_rl_energy_time(self):
        self.rl_gpu_energy = []
        self.rl_latency = []

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,

            threshold: float = None,
            measure_rl_energy: bool = False,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
                use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        exit_layer = None
        policy_states = []
        curr_time = 0
        curr_energy = 0

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  #

            if self.mode == "train_rl" or self.mode == "train_rl_gen":
                hidden_states_copy = hidden_states.clone()
                hidden_states_copy = self.norm(hidden_states_copy)
                logits = self.lm_head(hidden_states_copy).contiguous()
                state = self.extract_state(logits, hidden_states.clone(), idx)
                policy_states.append(state)
            if self.mode == "infer_ee" and idx in self.exit_indices:

                if measure_rl_energy:
                    self.monitor.begin_window("rl_agent_energy")

                action_logits = self.rl_model(hidden_states[0, -1, :])

                action_probabilities = F.softmax(action_logits / self.temperature, dim=0)

                action = torch.argmax(action_probabilities).item()

                if threshold is not None:
                    is_threshold_exceeded = action_probabilities[1] > threshold

                else:
                    is_threshold_exceeded = True
                if measure_rl_energy:
                    measurement = self.monitor.end_window("rl_agent_energy")

                    self.rl_gpu_energy.append(measurement.gpu_energy[0])
                    self.rl_latency.append(measurement.time)
                if action == 1 and is_threshold_exceeded:
                    exit_layer = idx
                    for i in range(idx + 1, len(self.layers)):
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)
                    break

                elif idx == self.num_hidden_layers - 1:
                    break
                else:
                    continue
        hidden_states = self.norm(hidden_states)

        if self.mode == "finetune":
            all_hidden_states = list(all_hidden_states)
            for idx in self.exit_indices:
                all_hidden_states[idx + 1] = self.norm(all_hidden_states[idx + 1])

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastExitAndStates(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            exit_layer=exit_layer,
            states=policy_states,  # used for RL training,
            has_exited=False
        )

    def extract_state(self, logits, hidden_state, current_layer):

        props = F.softmax(logits, dim=-1)
        max_probs, max_indices = torch.max(props, dim=-1)
        max_probs = max_probs[:, -1]

        max_indices = max_indices[:, -1]
        layer_info = torch.tensor([current_layer / (self.num_hidden_layers - 1)], device=logits.device)
        state = {
            "state": torch.cat([max_probs, layer_info], dim=-1),
            "hidden_state": self.norm(hidden_state),
            "prediction": torch.tensor([max_indices[0]], device=logits.device),
        }

        return state


class LlamaEESingleHeadRL(LlamaForCausalLM):

    def __init__(self, config, exit_indices: List[int] = None, mode="infer",
                 threshold: float = 0.5, temperature: float = 1,
                 rl_model: SelfBaseAlgorithm = None, measure_rl_energy: bool = False):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.exit_indices = exit_indices
        self.model = LlamaModelWithExit(config, exit_indices=exit_indices, mode=mode, lm_head=self.lm_head,
                                        temperature=temperature)
        self.exited_on = []
        self.mode = mode
        self.threshold = threshold
        self.hidden_size = config.hidden_size
        self.rl_model = rl_model
        self.logit_layer_list = []
        self.measure_rl_energy = measure_rl_energy
        self.states_list = []
        self.post_init()

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None, num_logits_to_keep: int = 0,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            threshold=self.threshold,
            measure_rl_energy=self.measure_rl_energy
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if self.mode == 'finetune':
            all_hidden_states = outputs.hidden_states
            loss_fct = CrossEntropyLoss()
            weight = 1 / (len(self.exit_indices) + 1)  # same weight for all exit indices + 1 because of the final layer
            for exit_index in self.exit_indices:
                # exit_index + 1, because the hidden states are shifted by one since the first hidden state is the
                # input (i.e. hidden_states = inputs_embeds + pos_embeds) i.e., in total there are 13 hidden states (
                # input_embeds+pos_embeds, 12 hidden states from the decoder) we apply layernorm to all hidden states
                # in the decoder, see above, to achieve the same behavior as with the original/last hidden-state
                hidden_state = all_hidden_states[exit_index + 1]
                logits = self.lm_head(hidden_state).contiguous()
                labels = labels.to(logits.device)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if loss is None:
                    loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
                else:
                    loss += loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight

            # final layer
            logits = self.lm_head(outputs[0]).contiguous()
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if loss is None:
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
            else:
                loss += loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
        elif self.mode == "train" and labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        elif self.mode == "train_rl":
            return outputs.states
        elif self.mode == "train_rl_gen":
            self.states_list.append(outputs.states)
        elif self.mode == "infer_ee":
            if outputs.exit_layer is not None:
                logits = self.lm_head(outputs[0]).contiguous()  # outputs[0] is hidden_state of last exit
                self.exited_on.append(outputs.exit_layer)
            else:
                logits = self.lm_head(outputs[0]).contiguous()
                self.exited_on.append(27)

        elif self.mode == "return_layer_logits":
            logit_layer_list = []
            for i, hidden_state in enumerate(outputs.hidden_states):
                if i == 0:
                    continue

                logits = self.lm_head(hidden_state).contiguous()
                logit_layer_list.append(logits)

            return logit_layer_list
        elif self.mode == "return_layer_logits_gen":
            logit_layer_list = []
            for i, hidden_state in enumerate(outputs.hidden_states):
                if i == 0:
                    continue

                logits = self.lm_head(hidden_state).contiguous()
                logit_layer_list.append(logits)
            self.logit_layer_list.append(logit_layer_list)



        elif self.mode == "random":
            pass
        elif self.mode != "infer" and self.mode != "infer_ee" and self.mode != "return_layer_logits_gen" and self.mode != "train_rl_gen":
            raise ValueError(f"mode {self.mode} not supported")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_exit_layers(self):
        return self.exited_on

    def clear_exit_layers(self):
        self.exited_on = []

    def get_logit_layer_list(self):
        return self.logit_layer_list

    def clear_logit_layer_list(self):
        self.logit_layer_list = []

    def update_threshold(self, thresh: float):
        self.threshold = thresh
        return self.threshold

    def get_states(self):
        return self.states_list

    def reset_states(self):
        self.states_list = []

    def get_rl_energy(self):
        return self.model.get_rl_energy()
