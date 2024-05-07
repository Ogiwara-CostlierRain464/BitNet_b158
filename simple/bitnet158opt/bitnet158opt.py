import torch
from torch import nn, Tensor
from typing import Tuple
from ..common import BitRMSNorm, LN


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

    def sign(self, x: torch.Tensor):
        return (x > 0).to(torch.int8) * 2 - 1

    def weight_quant(self, w: torch.Tensor) -> tuple[Tensor, float]:
        alpha = w.mean()
        w_center = w - alpha
        w_b = self.sign(w_center)
        beta = w.abs().mean()

        w_scaled = w_center / w_center.abs().max().clamp(min=self.EPS)
        # w_b = (w_b - w_center).detach() + w_center
        w_b = (w_b - w_scaled).detach() + w_scaled
        return w_b, beta


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
