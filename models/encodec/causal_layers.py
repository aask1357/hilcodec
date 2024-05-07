import typing as tp
import math

import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def initialize_cache(self, x: Tensor) -> Tensor:
        return torch.zeros(
            x.size(0), self.in_channels, self.causal_padding, device=x.device)

    def forward(self, x: Tensor, cache: Tensor) -> tp.Tuple[Tensor, Tensor]:
        x = torch.cat((cache, x), dim=2)
        cache = x[:, :, -self.causal_padding:]
        y = F.conv1d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return y, cache


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        receptive_field = self.dilation[0] * (self.kernel_size[0] - 1)
        self.causal_padding = receptive_field // self.stride[0]
        self.padding = (self.causal_padding * self.stride[0],)
        self.output_padding = (self.stride[0] - 1 + self.padding[0] - receptive_field,)
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def initialize_cache(self, x: Tensor) -> Tensor:
        return torch.zeros(
            x.size(0), self.in_channels, self.causal_padding, device=x.device)
    
    def forward(self, x: Tensor, cache: Tensor) -> tp.Tuple[Tensor, Tensor]:
        x = torch.cat([cache, x], dim=2)
        cache = x[:, :, -self.causal_padding:]
        y = F.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding,
                               self.output_padding, self.groups, self.dilation)
        return y, cache


def SConv1d(
    in_channels: int, out_channels: int,
    kernel_size: int, stride: int = 1, dilation: int = 1,
    groups: int = 1, bias: bool = True,
    norm: str = 'weight_norm', norm_kwargs: dict = {},
) -> nn.Module:
    _Conv = nn.Conv1d if kernel_size == 1 else CausalConv1d
    conv = _Conv(in_channels, out_channels, kernel_size, stride,
                 dilation=dilation, groups=groups, bias=bias)
    if norm == 'weight_norm':
        conv = weight_norm(conv, **norm_kwargs)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return conv


def SConvTranspose1d(
    in_channels: int, out_channels: int,
    kernel_size: int, stride: int = 1, dilation: int = 1,
    groups: int = 1, bias: bool = True,
    norm: str = 'weight_norm', norm_kwargs: dict = {},
) -> nn.Module:
    _Conv = nn.ConvTranspose1d if kernel_size == 1 else CausalConvTranspose1d
    conv = _Conv(in_channels, out_channels, kernel_size, stride,
                 dilation=dilation, groups=groups, bias=bias)
    if norm == 'weight_norm':
        conv = weight_norm(conv, **norm_kwargs)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return conv


class SLSTM(nn.LSTM):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__(dimension, dimension, num_layers)
        self.skip = skip
    
    def initialize_cache(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        h = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )
        return (h, h.clone())

    def forward(self, x: Tensor, cache: tp.Tuple[Tensor, Tensor]) -> tp.Tuple[Tensor, tp.Tuple[Tensor, Tensor]]:
        # x: [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        y, cache_out = super().forward(x, cache)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y, cache_out