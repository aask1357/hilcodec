import typing as tp
import math

import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def calculate_window_square_inverse(window: Tensor, N: int, hop_size: int) -> Tensor:
    # window: [N]
    T = (N + hop_size - 1) // hop_size      # <=> ceil(N / hop_size)
    window_square = window.square().view(-1, 1).expand(-1, T)
    window_square = F.fold(
        window_square,
        output_size = (1, hop_size*T + (N-hop_size)),
        kernel_size = (1, N),
        stride = (1, hop_size),
        padding = (0, 0)
    ).view(1, -1)    # [1, hop_size * T + (N-hop_size)]
    window_square = window_square[:, -N:-N+hop_size]  # [1, hop_size]
    assert torch.all(torch.ne(window_square, 0.0))
    return torch.reciprocal(window_square)  # 1 / window_square


class STFT(jit.ScriptModule):
# class STFT(nn.Module):
    '''Short-Time Fourier Transform
    forward(x):
        x: [B, 1, hop_size*L] or [B, hop_size*L]
        output: [B, N//2+1, L, 2]
    inverse(x):
        x: [B, N//2+1, L, 2]
        output: [B, 1, hop_size*L]'''

    __constants__ = ["n_fft", "hop_size", "cache_len", "norm"]

    def __init__(self, n_fft: int, hop_size: int, win_size: tp.Optional[int] = None,
                 win_type: tp.Optional[str] = "hann", window: tp.Optional[Tensor] = None,
                 norm: tp.Optional[str] = "backward",  # "forward" / "backward" / "ortho"
                 magnitude: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.cache_len = n_fft - hop_size
        self.norm = norm
        self.magnitude = magnitude

        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
            if win_size < n_fft:
                padding = n_fft - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(n_fft, dtype=torch.float32, device=device)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
            if win_size < n_fft:
                padding = n_fft - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"
        self.register_buffer("window", window.to(**factory_kwargs))
        self.window: Tensor

        inverse_window_square = calculate_window_square_inverse(window, n_fft, hop_size)
        inverse_window_square = inverse_window_square.to(**factory_kwargs)
        self.register_buffer('inverse_window_square', inverse_window_square)    # [1, hop_size]
        self.inverse_window_square: Tensor
    
    def initialize_cache(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        cache_input = torch.zeros(x.size(0), self.cache_len, dtype=x.dtype, device=x.device)
        cache_output = torch.zeros(x.size(0), self.cache_len, dtype=x.dtype, device=x.device)
        return cache_input, cache_output

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, H*L + N-H] or [B, H*L + N-H] where H = hop_size
        # output: [B, N//2+1, L]
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, HL + N-H]
        x = x.unfold(size=self.n_fft, step=self.hop_size, dimension=1)  # [B, L, N]
        x = x * self.window.view(1, 1, -1)      # [B, L, N]
        x = torch.fft.rfft(x, norm=self.norm)   # [B, L, N//2+1]
        x = x.transpose(1, 2)       # [B, N//2+1, L]
        # x = torch.stft(x, self.n_fft, self.hop_size, self.n_fft, window=self.window,
        #                center=False, onesided=True, return_complex=True)
        if self.magnitude:
            # x = x.abs()
            x = torch.view_as_real(x)
            x = torch.linalg.norm(x, dim=-1)
        return x

    def causal_forward(self, x: Tensor, cache: Tensor) -> tp.Tuple[Tensor, Tensor]:
        # x: [B, 1, H*L] or [B, H*L] where H = hop_size
        # output: [B, N//2+1, L]
        if x.dim() == 3:
            x = x.squeeze(2)  # [B, HL]
        x = torch.cat([cache, x], dim=1)  # [B, N-H + HL]
        cache = x[:, -self.cache_len:]    # [B, N-H]
        x = x.unfold(size=self.n_fft, step=self.hop_size, dimension=1)  # [B, L, N]
        x = x * self.window.view(1, 1, -1)      # [B, L, N]
        x = torch.fft.rfft(x, norm=self.norm)   # [B, L, N//2+1]
        x = x.transpose(1, 2)       # [B, N//2+1, L]
        if self.magnitude:
            x = x.abs()
        return x, cache

    def inverse(self, x: Tensor, cache_output: Tensor) -> tp.Tuple[Tensor, Tensor]:
        '''x: [B, N//2+1, L] / cache_output: [B, N-H] where H = hop_size
        output: [B, H*L]'''
        x = torch.fft.irfft(x, norm=self.norm, dim=1)  # [B, N, L]
        x = x * self.window.view(1, -1, 1)             # [B, N, L]

        # overlap-and-add
        L = x.size(2)
        x = torch.nn.functional.fold(
            x,
            output_size=(1, L * self.hop_size + self.cache_len),
            kernel_size=(1, self.n_fft),
            stride=self.hop_size
        ).squeeze(1).squeeze(1)         # [B, N-H + H*L]
        x[:, :self.cache_len] += cache_output

        cache_output = x[:, -self.cache_len:]

        # output = wav / window**2
        x = x[:, :-self.cache_len] * self.inverse_window_square.repeat(1, L)
        return x, cache_output


class CausalSTFT(nn.Module):
# class STFT(nn.Module):
    '''Short-Time Fourier Transform
    Implemented with Conv1d since onnx currently doesn't support torch.fft.rfft
    forward(x):
        x: [B, 1, hop_size*L] or [B, hop_size*L]
        output: [B, N//2+1, L, 2]'''

    __constants__ = ["n_fft", "hop_size", "cache_len", "norm"]

    def __init__(self, n_fft: int, hop_size: int, win_size: tp.Optional[int] = None,
                 win_type: tp.Optional[str] = "hann", window: tp.Optional[Tensor] = None,
                 norm: tp.Optional[str] = "backward",  # "forward" / "backward" / "ortho"
                 magnitude: bool = True,
                 device=None, dtype=None):
        assert magnitude, "STFTOnnx only supports magnitude=True"
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.cache_len = n_fft - 1
        self.norm = norm
        self.magnitude = magnitude

        if dtype is None:
            dtype = torch.float32
        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
            if win_size < n_fft:
                padding = n_fft - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(n_fft, **factory_kwargs)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
            if win_size < n_fft:
                padding = n_fft - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"
        
        n = torch.arange(n_fft, **factory_kwargs).view(1, 1, n_fft)
        k = torch.arange(n_fft//2+1, **factory_kwargs).view(-1, 1, 1)
        cos = torch.cos(-2*math.pi/n_fft*k*n)
        sin = torch.sin(-2*math.pi/n_fft*k*n)
        weight = torch.cat([cos, sin], dim=0) * window
        if norm == "forward":
            weight /= n_fft
        elif norm == "backward":
            pass
        elif norm == "ortho":
            weight /= math.sqrt(n_fft)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.register_buffer("weight", weight)
        self.weight: Tensor
    
    def initialize_cache(self, x: Tensor) -> Tensor:
        cache = torch.zeros(x.size(0), self.cache_len, dtype=x.dtype, device=x.device)
        return cache

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, H*L]
        # output: [B, N//2+1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, HL + N-H]
        x = F.conv1d(x, self.weight, None, stride=self.hop_size)
        B, C, T = x.shape
        x = x.view(B, 2, C//2, T)
        x = x.square().sum(dim=1).sqrt()
        return x


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
    norm: str = 'weight_norm'
) -> nn.Module:
    _Conv = nn.Conv1d if kernel_size == 1 else CausalConv1d
    conv = _Conv(in_channels, out_channels, kernel_size, stride,
                 dilation=dilation, groups=groups, bias=bias)
    if norm == 'weight_norm':
        conv = weight_norm(conv)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return conv


def SConvTranspose1d(
    in_channels: int, out_channels: int,
    kernel_size: int, stride: int = 1, dilation: int = 1,
    groups: int = 1, bias: bool = True,
    norm: str = 'weight_norm',
) -> nn.Module:
    _Conv = nn.ConvTranspose1d if kernel_size == 1 else CausalConvTranspose1d
    conv = _Conv(in_channels, out_channels, kernel_size, stride,
                 dilation=dilation, groups=groups, bias=bias)
    if norm == 'weight_norm':
        conv = weight_norm(conv)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return conv
