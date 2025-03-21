from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import OPTConfig, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, \
    CausalLMOutputWithPast
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer, OPTModel, \
    OPTForCausalLM
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)


class OPTAttentionWithExit(OPTAttention):

    def __init__(self, config: OPTConfig, is_decoder):
        super().__init__(config, is_decoder)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            just_propagate: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        if just_propagate:
            return past_key_value, None, None

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayerWithExit(OPTDecoderLayer):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.self_attn = OPTAttentionWithExit(config, is_decoder=True)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            just_propagate: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            just_propagate=just_propagate
        )

        if just_propagate:
            return hidden_states  # those are ONLY the past_key_value, see return statement of self_attn

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@dataclass
class BaseModelOutputWithPastAndExitLayer(BaseModelOutputWithPast):
    exit_layer: int = None


class OPTDecoderWithExit(OPTDecoder):
    def __init__(self, config: OPTConfig, exit_indices: List[int], mode: str):
        super().__init__(config)
        self.layers = nn.ModuleList([OPTDecoderLayerWithExit(config) for _ in range(config.num_hidden_layers)])
        self.exit_positions = exit_indices
        self.mode = mode
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            threshhold: Optional[float] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        exit_layer = None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # print(idx)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if not self.training and idx == exit_layer and self.mode == "infer_ee":
                exit_layer = idx
                for i in range(idx + 1, len(self.layers)):
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)
                break

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            if self.mode == "finetune":
                all_hidden_states = list(all_hidden_states)
                for idx in self.exit_positions:
                    all_hidden_states[idx + 1] = self.final_layer_norm(all_hidden_states[idx + 1])

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
            if self.mode == "finetune":
                all_hidden_states = list(all_hidden_states)
                for idx in self.exit_positions:
                    all_hidden_states[idx] = self.project_out(all_hidden_states[idx])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndExitLayer(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            exit_layer=exit_layer
        )


class OPTModelWithExit(OPTModel):

    def __init__(self, config: OPTConfig, exit_indices: List[int], mode: str):
        super().__init__(config)
        self.decoder = OPTDecoderWithExit(config, exit_indices, mode)
        self.exit_indices = exit_indices
        self.post_init()

    # Does not need to use the forward method from OPTModel, since we are using the one with LMHead
    # (OPTForCausalLM).


class OPTEESingleHead(OPTForCausalLM):

    def __init__(self, config: OPTConfig, threshhold: float = 0.99, mode='infer', exit_indices: List[int] = None,
                 exits_first_half: List[int] = None,
                 exits_second_half: List[int] = None, weight_mode="uniform", log_loss=False,
                 total_weight_first_half=0.6, total_weight_second_half=0.2):
        super().__init__(config)

        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.weight_mode = weight_mode
        self.exit_first_half = exits_first_half
        self.exit_second_half = exits_second_half
        self.log_loss = log_loss
        self.threshhold = threshhold
        self.exit_indices = exit_indices
        self.model = OPTModelWithExit(config, exit_indices, mode)
        if self.weight_mode == "geometric" and mode == "finetune":
            if self.exit_first_half is None or self.exit_second_half is None:
                raise Exception("need to specify layers for finetuning geometric")
            weights = self.get_weights(total_weight_first_half, len(self.exit_first_half), total_weight_second_half,
                                       len(self.exit_second_half))
            self.weights = {}
            print(len(exit_indices))
            for i, w in enumerate(weights):
                if i < len(exit_indices):
                    self.weights[exit_indices[i]] = w
                else:
                    self.weights[config.num_hidden_layers - 1] = weights[i]  # final layer
            if sum(weights) != 1.0:
                raise Exception("Weights do not sum up to 1.0")
            print(self.weights)
        self.mode = mode
        self.num_hidden_layers = config.num_hidden_layers
        self.post_init()

    def tie_weights_inter_lm(self):
        # Tie the weights of lm_head to all intermediate_lm_heads
        for head in self.intermediate_lm_heads:
            head.weight = nn.Parameter(self.lm_head.weight.data.clone())

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            position_ids: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn, exit_layer)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            threshhold=self.threshhold
        )
        loss = None

        logits = self.lm_head(outputs[0]).contiguous()

        if self.mode == 'finetune':
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

        elif self.mode == "infer" and labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        elif self.mode == "train_rl":
            pass
        elif self.mode == "infer_ee" and outputs.exit_layer is not None:
            print(outputs.exit_layer)

        elif self.mode == "return_layer_logits":
            logit_layer_list = []
            for i, hidden_state in enumerate(outputs.hidden_states):
                # skip first hidden state, because it is the input
                if i == 0:
                    continue

                logits = self.lm_head(hidden_state).contiguous()
                logit_layer_list.append(logits.cpu())

            return logit_layer_list

        elif self.mode != "infer":
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
