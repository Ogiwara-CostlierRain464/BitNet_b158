import torch
from torch import nn, Tensor


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    # maxが0に近い値になる可能性があるので、clamp
    scale: float = 128 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp(-128, 127)


def sign(x: torch.Tensor) -> torch.Tensor:
    # 1 * 2 - 1 = 1, 0 * 2 - 1 = -1
    return (x > 0).to(torch.int8) * 2 - 1


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    alpha = w.mean()
    return sign(w - alpha)


def LN(x: torch.Tensor) -> torch.Tensor:
    e_x = x.mean()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x - e_x) / torch.rsqrt(var + 1e-5)


def activation_quant2(x: torch.Tensor, before_activation_func=True) -> torch.Tensor:
    if before_activation_func:
        eta = x.min()
        scale: float = 128 / (x - eta).abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return ((x - eta) * scale).round().clamp(0, 127)
    else:
        scale: float = 128 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (x * scale).round().clamp(-128, 127)

