import torch
from torch import nn, Tensor
from typing import Tuple
from ..common import BitRMSNorm, LN

EPS = 1e-5
Qb = 2 ** 7


def activation_quant(x: torch.Tensor) -> Tensor:
    scale: float = Qb / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=EPS)  # overflow防止
    y = (x * scale).round().clamp(-Qb, Qb - 1) / scale

    return y


def weight_quant(w: torch.Tensor) -> tuple[Tensor, float]:
    scale = 1.0 / w.abs().mean().clamp_(min=EPS)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinear158(nn.Linear):
    """
    MSRの実装に合わせ、STEをactivation_quantの外で実行する
    """
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear158, self).__init__(in_features, out_features, bias)
        self.layer_norm = BitRMSNorm(hidden_size=in_features, eps=EPS)
        # self.layer_norm = LN() # BitNet1の元論文

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.layer_norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = torch.nn.functional.linear(x_quant, w_quant, self.bias)
        return y
