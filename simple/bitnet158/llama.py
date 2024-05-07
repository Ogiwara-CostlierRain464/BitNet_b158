import warnings
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
)
from .bitnet158 import BitLinear158
import torch
from torch import nn


class BitLlamaConfig(LlamaConfig):
    model_type = "bit_llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BitLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = BitLinear158(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BitLinear158(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BitLinear158(self.intermediate_size, self.hidden_size, bias=False)


class BitLlamaAttention(LlamaAttention):
    def __init__(self, config: BitLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = BitLinear158(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear158(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear158(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = BitLinear158(self.hidden_size, self.hidden_size, bias=False)


class BitLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: BitLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = BitLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = BitLlamaMLP(config)
        del self.input_layernorm
        del self.post_attention_layernorm

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        refers: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L693
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BitLlamaModel(LlamaModel):
    config_class = BitLlamaConfig

    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [BitLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class BitLlamaForCausalLM(LlamaForCausalLM):
    config_class = BitLlamaConfig

    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.model = BitLlamaModel(config)
        self.lm_head = BitLinear158(config.hidden_size, config.vocab_size, bias=False)
