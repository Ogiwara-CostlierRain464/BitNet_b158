import torch
from torch import nn, Tensor
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


class BitLinear158Opt(nn.Linear):
    """
    STEをactivation_quantの中で実行し、最適化する。
    """
    EPS = 1e-6
    Qb = 2 ** 7

    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear158Opt, self).__init__(in_features, out_features, bias)
        self.layer_norm = BitRMSNorm(hidden_size=in_features, eps=self.eps)
        # self.layer_norm = LN() # BitNet1の元論文

    def activation_quant(self, x: torch.Tensor) -> tuple[Tensor, float]:
        eps = self.EPS
        Qb = self.Qb

        gamma: float = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=eps)  # overflow防止
        x_scaled = x * Qb / gamma
        x_q = x_scaled.round().clamp(-Qb, Qb - 1)

        # STE
        x_q = (x_q - x_scaled).detach() + x_scaled
        return x_q, gamma

    def weight_quant(self, w: torch.Tensor) -> tuple[Tensor, float]:
        gamma = w.abs().mean().clamp(min=self.EPS)
        w_tr = (w / gamma).round().clamp_(-1, 1)
        # STE
        w_tr = (w_tr - w).detach() + w
        return w_tr, gamma

    def forward(self, x: Tensor) -> Tensor:
        # LayerNorm
        x_norm = self.layer_norm(x)
        # Absmax Quantization
        x_q, gamma = self.activation_quant(x_norm)
        # 1-bit weights
        w_q, beta = self.weight_quant(self.weight)
        # ⊗
        matmul = torch.nn.functional.linear(x_q, w_q, self.bias)
        # Dequantization
        y = matmul * (beta * gamma / self.Qb)

        return y
