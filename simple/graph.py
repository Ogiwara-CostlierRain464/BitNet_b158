import torch
import torch.nn as nn
from torch import Tensor
from torchviz import make_dot

EPS = 1e-5


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


def activation_quant(x: torch.Tensor) -> Tensor:
    scale: float = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=EPS)  # overflow防止
    y = (x * scale).round().clamp(-128, 127) / scale
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

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.layer_norm(x)
        #x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        #w_quant = w + (weight_quant(w) - w).detach()
        x_quant = activation_quant(x_norm)
        w_quant = weight_quant(w)
        y = torch.nn.functional.linear(x_quant, w_quant, self.bias)
        return y



if __name__ == '__main__':
    m = BitLinear158(2, 2)
    data = torch.randn(2, 2)
    out = m(data)

    image = make_dot(out, params=dict(m.named_parameters()))
    image.format = "png"
    image.render("BitNet")