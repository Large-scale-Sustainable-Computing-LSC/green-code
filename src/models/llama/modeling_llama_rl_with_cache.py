from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pynvml
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from torch import nn
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM, LlamaConfig, Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, LlamaAttention, LlamaSdpaAttention
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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaSdpaAttentionWithCache(LlamaSdpaAttention):

    def get_key_value(self, hidden_states, position_embeddings, position_ids, cache_pos):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return key_states, value_states, {"sin": sin, "cos": cos, "cache_position": cache_pos}

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            prop=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if prop:
                return None, None, past_key_value

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class LlamaDecoderLayerWithCache(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaSdpaAttentionWithCache(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def do_layer_norm(self, hidden_state):
        return self.input_layernorm(hidden_state)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            prop=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            prop=prop,
        )
        if prop:
            return present_key_value
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModelWithExit(LlamaModel):

    def __init__(self, config: LlamaConfig, mode: str, exit_indices: List[int] = None, lm_head=None, temperature=1):
        super().__init__(config)
        self.exit_indices = exit_indices
        self.mode = mode
        self.lm_head = lm_head
        self.temprature = temperature
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerWithCache(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rl_model = None
        self.sb3_model = None

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.rl_gpu_energy = []

    def set_RL_model(self, rl_model: SelfBaseAlgorithm, mode="PPO"):
        self.rl_model = SB3TorchWrapper(rl_model, mode=mode).to("cuda")
        self.sb3_model = rl_model

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
            is_first_token: bool = False,

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
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #  if use_cache and isinstance(past_key_values, Cache):
        #       past_key_values = DynamicCache.to_legacy_cache(past_key_values)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
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

            if self.mode == "train_rl":
                hidden_states_copy = hidden_states.clone()
                hidden_states_copy = self.norm(hidden_states_copy)
                logits = self.lm_head(hidden_states_copy).contiguous()
                state = self.extract_state(logits, hidden_states, idx)
                policy_states.append(state)

                temperature = 1

                action_logits = self.rl_model(hidden_states[0, -1, :])
                action_probabilities = F.softmax(action_logits / temperature, dim=0)

                action = torch.argmax(action_probabilities).item()
                if threshold is not None:
                    is_threshold_exceeded = action_probabilities[1] > threshold
                else:
                    is_threshold_exceeded = True
                measure = self.monitor.end_window("RL", sync_execution=True)
                if action == 1 and is_threshold_exceeded:
                    exit_layer = idx
                    self.monitor.begin_window("KV", sync_execution=True)
                    for i in range(idx + 1, len(self.layers)):
                        # if output_hidden_states:
                        #    all_hidden_states += (hidden_states,)
                        if use_cache:
                            next_decoder_cache = self.layers[i](hidden_states, past_key_value=past_key_values,
                                                                position_ids=position_ids,
                                                                cache_position=cache_position, prop=True)

                    measure = self.monitor.end_window("KV", sync_execution=True)
                    break


                elif idx == self.num_hidden_layers - 1:
                    break
                else:
                    continue
            if self.mode == "infer_ee" and idx in self.exit_indices and not is_first_token:

                action_logits = self.rl_model(hidden_states[0, -1, :])
                action_probabilities = F.softmax(action_logits / self.temprature, dim=0)
                action = torch.argmax(action_probabilities).item()

                if action == 1:

                    if action_probabilities[1] < threshold:
                        continue
                    exit_layer = idx
                    for i in range(idx + 1, len(self.layers)):
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)
                        if use_cache:
                            next_decoder_cache = self.layers[i](hidden_states, past_key_value=past_key_values,
                                                                position_ids=position_ids,
                                                                cache_position=cache_position, prop=True)
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
            "hidden_state": hidden_state,
            "prediction": torch.tensor([max_indices[0]], device=logits.device),
        }

        return state


class LlamaEESingleHeadRLCaching(LlamaForCausalLM):

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
        self.current_tk = 0
        self.threshold = threshold
        self.hidden_size = config.hidden_size
        self.rl_model = rl_model
        self.measure_rl_energy = measure_rl_energy
        # Initialize weights and apply final processing
        self.post_init()

    def get_hidden_size(self):
        return self.hidden_size

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if input_ids.shape == torch.Size([1, 1]):
            is_first = False

        else:
            is_first = True

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
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            threshold=self.threshold,
            measure_rl_energy=self.measure_rl_energy,
            is_first_token=is_first
        )

        hidden_states = outputs[0]

        if self.mode != "infer_ee":
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:

                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if self.mode == 'finetune':
            #  total_weight = 0
            # iterate through exit_indices
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
        elif self.mode == "infer_ee":
            if outputs.exit_layer is not None:
                logits = self.lm_head(outputs[0]).contiguous()  # outputs[0] is hidden_state of last exit
                self.exited_on.append(outputs.exit_layer)
            else:
                logits = self.lm_head(outputs[0]).contiguous()
                self.exited_on.append(27)
            # print(outputs.exit_layer)

        elif self.mode == "return_layer_logits":
            logit_layer_list = []
            for i, hidden_state in enumerate(outputs.hidden_states):
                # skip first hidden state, because it is the input
                if i == 0:
                    continue

                logits = self.lm_head(hidden_state).contiguous()
                logit_layer_list.append(logits)

            return logit_layer_list
        elif self.mode == "random":
            pass
        elif self.mode != "infer" and self.mode != "infer_ee" and self.mode != "measure_overhead":
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
