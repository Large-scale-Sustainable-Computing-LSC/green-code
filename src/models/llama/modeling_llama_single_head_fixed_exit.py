from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM, LlamaConfig, Cache, DynamicCache, \
    AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


@dataclass
class BaseModelOutputWithPastAndExitLayer(BaseModelOutputWithPast):
    exit_layer: int = None


class LlamaModelWithExit(LlamaModel):

    def __init__(self, config: LlamaConfig, exit_indices: List[int] = None, mode: str = "infer"):
        super().__init__(config)
        self.exit_indices = exit_indices
        self.mode = mode

        # Initialize weights and apply final processing
        self.post_init()

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
            exit_index: int = None
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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

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

            if idx == exit_index and self.mode == "infer_ee" and not self.training:

                for i in range(idx + 1, len(self.layers)):
                    all_hidden_states += (hidden_states,)
                break

        hidden_states = self.norm(hidden_states)

        if self.mode == "finetune":
            all_hidden_states = list(all_hidden_states)
            for idx in self.exit_indices:
                all_hidden_states[idx + 1] = self.norm(all_hidden_states[idx + 1])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndExitLayer(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            exit_layer=exit_index
        )


class LlamaEESingleHead(LlamaForCausalLM):

    def __init__(self, config, exit_index: int = 1, weight_mode="uniform", exit_indices: List[int] = None, mode="infer",
                 exits_first_half: List[int] = None,
                 exits_second_half: List[int] = None, log_loss=False):
        super().__init__(config)
        self.model = LlamaModelWithExit(config, exit_indices=exit_indices, mode=mode)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.exit_indices = exit_indices
        self.mode = mode
        self.exit_first_half = exits_first_half
        self.exit_second_half = exits_second_half
        self.weight_mode = weight_mode
        self.exit_index = exit_index
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.log_loss = log_loss
        if self.weight_mode == "geometric" and mode == "finetune":
            if self.exit_first_half is None or self.exit_second_half is None:
                raise Exception("need to specify layers for finetuning geometric")
            weights = self.get_weights(0.7, len(self.exit_first_half), 0.2, len(self.exit_second_half))
            self.weights = {}
            print(len(exit_indices))
            for i, w in enumerate(weights):
                if i < len(exit_indices):
                    self.weights[exit_indices[i]] = w
                else:
                    self.weights[config.num_hidden_layers - 1] = weights[i]  # final layer
            if sum(weights) != 1.0:
                raise Exception("Weights do not sum up to 1.0")
        if self.mode == "finetune" and log_loss:
            self.log_data = []

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
            exit_index=self.exit_index
        )

        hidden_states = outputs[0]
        if self.mode == "infer" or self.mode == "infer_ee":
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:

                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if self.mode == "infer" and labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        elif self.mode == "return_layer_logits":
            logit_layer_list = []
            for i, hidden_state in enumerate(outputs.hidden_states):
                # skip first hidden state, because it is the input
                if i == 0:
                    continue

                logits = self.lm_head(hidden_state).contiguous()
                logit_layer_list.append(logits)

            return logit_layer_list
        elif self.mode == 'finetune':

            if self.weight_mode == "uniform":
                weight = 1 / (
                        len(self.exit_indices) + 1)  # same weight for all exit indices + 1 because of the final layer

            elif self.weight_mode == "geometric" and self.weights is None:
                raise Exception("Unsupported weight mode")

            #  total_weight = 0
            # iterate through exit_indices
            all_hidden_states = outputs.hidden_states
            loss_fct = CrossEntropyLoss()
            for exit_index in self.exit_indices:
                # exit_index + 1, because the hidden states are shifted by one since the first hidden state is the
                # input (i.e. hidden_states = inputs_embeds + pos_embeds) e.g. (12 layer model)., in total there are 13 hidden states (
                # input_embeds+pos_embeds, 12 hidden states from the decoder) we apply layernorm to all hidden states
                # in the decoder, see above, to achieve the same behavior as with the original/last hidden-state
                hidden_state = all_hidden_states[exit_index + 1]
                logits = self.lm_head(hidden_state).contiguous()
                labels = labels.to(logits.device)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if self.weight_mode == "geometric":
                    weight = self.weights[exit_index]

                if loss is None:
                    loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
                    if self.log_loss:
                        self.log_data.append([exit_index, loss.item()])
                else:
                    loss_value = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
                    if self.log_loss:
                        self.log_data.append([exit_index, loss_value.item()])
                    loss += loss_value
            if self.weight_mode == "geometric":
                weight = self.weights[self.num_hidden_layers - 1]
            # final layer
            logits = self.lm_head(outputs[0]).contiguous()
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if loss is None:
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
                if self.log_loss:
                    self.log_data.append([self.num_hidden_layers - 1, loss.item()])
            else:
                loss_value = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)) * weight
                if self.log_loss:
                    self.log_data.append([self.num_hidden_layers - 1, loss_value.item()])
                loss += loss_value

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

    def get_weights(self, total_weight_first_half, num_layers_first_half, total_weight_second_half,
                    num_layers_second_half,
                    include_final_layer=True,
                    ratio_first_half=0.9, ratio_second_half=0.9):

        shares_X = [ratio_first_half ** i for i in range(num_layers_first_half)]
        total_X = sum(shares_X)
        normalized_shares_X = [(share / total_X) * total_weight_first_half for share in shares_X]

        shares_Y = [ratio_second_half ** i for i in range(num_layers_second_half)]
        total_Y = sum(shares_Y)
        normalized_shares_Y = [(share / total_Y) * total_weight_second_half for share in shares_Y]

        combined_shares = normalized_shares_X + normalized_shares_Y
        if include_final_layer:
            combined_shares.append(1 - np.sum(combined_shares))

        return combined_shares
