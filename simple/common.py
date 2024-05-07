import torch
from torch import nn
from typing import Tuple


class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BitRMSNorm is equivalent to LlamaRMSNorm and T5LayerNorm
        refers: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L76C1-L90C59
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# BitNet1ではSubLNが採用されている。
class LN(nn.Module):
    eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_x = x.mean()
        up = x - e_x
        var = x.pow(2).mean(-1, keepdim=True)
        down = torch.rsqrt(var + self.eps)
        return up / down
