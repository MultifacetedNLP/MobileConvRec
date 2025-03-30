from collections import defaultdict
from typing import Optional
import torch.nn as nn
import torch
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention,T5LayerCrossAttention,T5LayerFF

class Combined_Block_2a(nn.Module):
    def __init__(self, config,config2,block, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            block.layer[0]
        )
        self.layer.append(block.layer[1])
        self.layer.append(T5LayerCrossAttention(config2, layer_idx=layer_idx))

        self.layer.append(block.layer[2])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_statesA=None,
        encoder_attention_maskA=None,
        encoder_hidden_statesB=None,
        encoder_attention_maskB=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
    ):
        
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        
        do_cross_attention = True
        if do_cross_attention:
            cross_attention_outputsA = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_statesA,
                attention_mask=encoder_attention_maskA,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputsA[:2]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputsA[2:]

    
        if do_cross_attention:
            cross_attention_outputsB = self.layer[2](
                hidden_states,
                key_value_states=encoder_hidden_statesB,
                attention_mask=encoder_attention_maskB,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputsB[:2]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputsB[2:]




        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (past_key_value,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
    

import torch
import torch.nn as nn
import warnings
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Block, T5LayerNorm, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.cache_utils import Cache, StaticCache, DynamicCache, EncoderDecoderCache
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.utils import is_torchdynamo_compiling, add_start_docstrings
from transformers.modeling_attn_mask_utils import AttentionMaskConverter




class Combined_Stack_2a(T5PreTrainedModel):
    def __init__(self, config,config2,stack):
        super().__init__(config)
        self.is_decoder = stack.is_decoder
        self.embed_tokens = stack.embed_tokens

        self.block = nn.ModuleList(
            [Combined_Block_2a(config,config2,stack.block[i] , layer_idx=i+len(stack.block)) for i in range(len(stack.block))]
        )
        self.final_layer_norm = stack.final_layer_norm
        self.dropout = stack.dropout

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_statesA=None,
        encoder_attention_maskA=None,
        encoder_hidden_statesB=None,
        encoder_attention_maskB=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_statesA is not None:
            encoder_batch_sizeA, encoder_sequence_lengthA, _ = encoder_hidden_statesA.size()
            encoder_hidden_shapeA = (encoder_batch_sizeA, encoder_sequence_lengthA)
            if encoder_attention_maskA is None:
                encoder_attention_maskA = torch.ones(
                    encoder_hidden_shapeA, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_maskA = self.invert_attention_mask(encoder_attention_maskA)
        else:
            encoder_extended_attention_maskA = None

        
        if self.is_decoder and encoder_hidden_statesB is not None:
            encoder_batch_sizeB, encoder_sequence_lengthB, _ = encoder_hidden_statesB.size()
            encoder_hidden_shapeB = (encoder_batch_sizeB, encoder_sequence_lengthB)
            if encoder_attention_maskB is None:
                encoder_attention_maskB = torch.ones(
                    encoder_hidden_shapeB, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_maskB = self.invert_attention_mask(encoder_attention_maskB)
        else:
            encoder_extended_attention_maskB = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_statesA is not None:
                    encoder_hidden_statesA = encoder_hidden_statesA.to(hidden_states.device)
                if encoder_extended_attention_maskA is not None:
                    encoder_extended_attention_maskA = encoder_extended_attention_maskA.to(hidden_states.device)
                if encoder_hidden_statesB is not None:
                    encoder_hidden_statesB = encoder_hidden_statesB.to(hidden_states.device)
                if encoder_extended_attention_maskB is not None:
                    encoder_extended_attention_maskB = encoder_extended_attention_maskB.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_statesA,
                    encoder_extended_attention_maskA,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_statesA=encoder_hidden_statesA,
                    encoder_attention_maskA=encoder_extended_attention_maskA,
                    encoder_hidden_statesB=encoder_hidden_statesB,
                    encoder_attention_maskB=encoder_extended_attention_maskB,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_statesA is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    



import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import is_torch_fx_proxy, T5LayerNorm
import copy





class Combined_Model_2a(nn.Module):
    def __init__(self, modelA,modelB):
        super().__init__()
        self.config = modelA.config
        self.config2 = modelB.config
        self.encoderA = modelA.encoder
        self.encoderB = modelB.encoder
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        decoder_config2 = copy.deepcopy(self.config2)
        decoder_config2.is_decoder = True
        decoder_config2.is_encoder_decoder = False
        decoder_config2.num_layers = self.config2.num_decoder_layers
        self.decoder = Combined_Stack_2a(decoder_config,decoder_config2, modelA.decoder)
        self.lm_head = modelA.lm_head



        




    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    


        
    def forward(self, input_idsA,attention_maskA,input_idsB,attention_maskB,labels=None):

        encoder_outputsA = self.encoderA(
                input_ids=input_idsA,
                attention_mask=attention_maskA
            )
        
        encoder_outputsB = self.encoderB(
                input_ids=input_idsB,
                attention_mask=attention_maskB
            )
        
        hidden_statesA = encoder_outputsA[0]
        hidden_statesB = encoder_outputsB[0]

        decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_statesA=hidden_statesA,
            encoder_attention_maskA=attention_maskA,
            encoder_hidden_statesB=hidden_statesB,
            encoder_attention_maskB=attention_maskB
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits
        )


    def generate(
        self,
        input_idsA: torch.Tensor,
        attention_maskA: torch.Tensor,
        input_idsB: torch.Tensor,
        attention_maskB: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 5,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs
    ):
        with torch.no_grad():
            # Encode inputs with both encoders
            encoder_outputsA = self.encoderA(input_idsA, attention_mask=attention_maskA)
            encoder_outputsB = self.encoderB(input_idsB, attention_mask=attention_maskB)
            hidden_statesA = encoder_outputsA[0]
            hidden_statesB = encoder_outputsB[0]

            # Initialize beams
            device = input_idsA.device
            batch_size = input_idsA.size(0)
            vocab_size = self.config.vocab_size
            
            # Initialize decoder input with start token
            input_ids = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=device
            )

            # Beam search initialization
            beam_scores = torch.zeros(batch_size, num_beams, device=device)
            beam_scores[:, 1:] = -1e9  # Force first beam selection
            beam_sequences = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
            finished = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)

            # Expand encoder outputs for beam search
            expanded_hiddenA = hidden_statesA.unsqueeze(1).repeat(1, num_beams, 1, 1)
            expanded_hiddenA = expanded_hiddenA.view(batch_size * num_beams, -1, hidden_statesA.size(-1))
            expanded_hiddenB = hidden_statesB.unsqueeze(1).repeat(1, num_beams, 1, 1)
            expanded_hiddenB = expanded_hiddenB.view(batch_size * num_beams, -1, hidden_statesB.size(-1))

            for step in range(max_length):
                # Flatten for processing
                flat_sequences = beam_sequences.view(batch_size * num_beams, -1)
                
                # Get decoder outputs
                decoder_outputs = self.decoder(
                    input_ids=flat_sequences,
                    encoder_hidden_statesA=expanded_hiddenA,
                    encoder_attention_maskA=attention_maskA.repeat_interleave(num_beams, dim=0),
                    encoder_hidden_statesB=expanded_hiddenB,
                    encoder_attention_maskB=attention_maskB.repeat_interleave(num_beams, dim=0)
                )
            
                # Get next token logits
                next_token_logits = self.lm_head(decoder_outputs[0][:, -1, :])
                next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                
                # Reshape to (batch_size, num_beams, vocab_size)
                next_token_probs = next_token_probs.view(batch_size, num_beams, -1)
                
                # Apply length penalty
                scores = beam_scores.unsqueeze(-1) + next_token_probs
                scores = scores / (step + 1) ** length_penalty

                # Handle n-gram repetition
                if no_repeat_ngram_size > 0:
                    banned_tokens = self._calc_banned_ngram_tokens(
                        flat_sequences, no_repeat_ngram_size, step + 1
                    )
                    banned_tokens = banned_tokens.view(batch_size, num_beams, -1)
                    for batch_idx in range(batch_size):
                        for beam_idx in range(num_beams):
                            scores[batch_idx, beam_idx, banned_tokens[batch_idx, beam_idx]] = -float("inf")
    
                # Get top candidates
                vocab_size = scores.size(-1)
                scores = scores.view(batch_size, -1)
                top_scores, top_indices = torch.topk(scores, num_beams, dim=1)
            
                # Update beam indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
    
                # Update sequences
                beam_sequences = torch.cat([
                    beam_sequences[torch.arange(batch_size).unsqueeze(-1), beam_indices],
                    token_indices.unsqueeze(-1)
                ], dim=-1)
            
            # Update scores
                beam_scores = top_scores
    
                # Check for EOS tokens
                eos_mask = token_indices == self.config.eos_token_id
                finished = finished | eos_mask
    
                # Early stopping if all beams finished
                if early_stopping and finished.all():
                    break

            # Select best sequences
            best_sequences = []
            for batch_idx in range(batch_size):
                batch_scores = beam_scores[batch_idx]
                best_idx = torch.argmax(batch_scores)
                best_sequences.append(beam_sequences[batch_idx, best_idx])

        return torch.stack(best_sequences)

    def _calc_banned_ngram_tokens(self, sequences, ngram_size, cur_len):
        """Calculate banned tokens for n-gram repetition prevention"""
        batch_size, seq_len = sequences.shape
        banned_tokens = [[] for _ in range(batch_size)]
        
        if seq_len < ngram_size or ngram_size == 0:
            return banned_tokens

        # For each sequence in batch
        for batch_idx in range(batch_size):
            sequence = sequences[batch_idx].tolist()
            generated_ngrams = defaultdict(list)
            
            # Create ngrams
            for ngram_start in range(seq_len - ngram_size + 1):
                ngram = tuple(sequence[ngram_start:ngram_start+ngram_size])
                generated_ngrams[ngram[:-1]].append(ngram[-1])

            # Get banned tokens for current position
            current_ngram = tuple(sequence[-ngram_size+1:])
            banned_tokens[batch_idx] = generated_ngrams.get(current_ngram, [])

        return banned_tokens
