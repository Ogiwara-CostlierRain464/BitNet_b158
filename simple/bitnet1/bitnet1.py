import torch
from torch import nn, Tensor
from typing import Tuple
from ..common import BitRMSNorm, LN

EPS = 1e-5
Qb = 2 ** 7


def activation_quant(x: torch.Tensor, before_linear=True) -> Tensor:
    if before_linear:
        # (4), (5)
        scale: float = Qb / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=EPS)  # overflow防止
        y = (x * scale).round().clamp(-Qb, Qb - 1) / scale
    else:
        eta = x.min()
        scale: float = Qb / (x - eta).abs().max(dim=-1, keepdim=True).values.clamp_(min=EPS)
        y = ((x - eta) * scale).round().clamp(0, Qb - 1) / scale

    return y


def sign(x: torch.Tensor):
    return (x > 0).to(torch.int8) * 2 - 1


def weight_quant(w: torch.Tensor) -> Tensor:
    alpha = w.mean()
    w_b = sign(w - alpha)
    beta = w.abs().mean()
    return w_b


class BitLinear1(nn.Linear):
    """
    MSRの実装に合わせ、STEをactivation_quantの外で実行する
    """
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear1, self).__init__(in_features, out_features, bias)
        self.layer_norm = BitRMSNorm(hidden_size=in_features, eps=EPS)
        # self.layer_norm = LN() # BitNet1の元論文

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.layer_norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = torch.nn.functional.linear(x_quant, w_quant, self.bias)
        return y
