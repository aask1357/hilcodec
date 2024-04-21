import warnings
from typing import Optional
import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from scipy.signal import kaiser


#class STDCT(torch.jit.ScriptModule):
class STDCT(nn.Module):
    '''Short-Time Discrete Cosine Transform II
    forward(x, inverse=False):
        x: [B, 1, hop_size*T] or [B, hop_size*T]
        output: [B, N, T+1] (center = True)
        output: [B, N, T]   (center = False)
    forward(x, inverse=True):
        x: [B, N, T+1] (center = True)
        x: [B, N, T]   (center = False)
        output: [B, 1, hop_size*T]'''

    __constants__ = ["N", "hop_size", "padding"]

    def __init__(self, N: int, hop_size: int, win_size: Optional[int] = None,
                 win_type: Optional[str] = "hann", center: bool = False,
                 window: Optional[Tensor] = None, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.hop_size = hop_size
        if center:
            self.padding = (N + 1) // 2     # <=> ceil{N / 2}
            self.output_padding = N % 2
        else:
            self.padding = (N - hop_size + 1) // 2  # <=> ceil{(N - hop_size) / 2}
            self.output_padding = (N - hop_size) % 2
            self.clip = (hop_size % 2 == 1)

        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = N
        
        if window is not None:
            win_size = window.size(-1)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(N, dtype=torch.float32, device=device)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert N >= win_size, f"N({N}) must be bigger than win_size({win_size})"
        n = torch.arange(N, dtype=torch.float32, device=device).view(1, 1, N)
        k = n.view(N, 1, 1)
        _filter = torch.cos(math.pi/N*k*(n+0.5)) * math.sqrt(2/N)
        _filter[0, 0, :] /= math.sqrt(2)
        dct_filter = (_filter * window.view(1, 1, N)).to(**factory_kwargs)
        window_square = window.square().view(1, -1, 1).to(**factory_kwargs)
        self.register_buffer('filter', dct_filter)
        self.register_buffer('window_square', window_square)
        self.filter: Tensor
        self.window_square: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, hop_size*T] or [B, hop_size*T]
        # output: [B, N, T+1] (center = True)
        # output: [B, N, T]   (center = False)
        if x.dim() == 2:
            x = x.unsqueeze(1)
    
        x = F.conv1d(x, self.filter, bias=None, stride=self.hop_size,
            padding=self.padding)
        if self.clip:
            x = x[:, :, :-1]
        return x
    
    @torch.jit.export
    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, N, T+1] (center = True)
        # x: [B, N, T]   (center = False)
        # output: [B, 1, hop_size*T]
        wav =  F.conv_transpose1d(spec, self.filter, bias=None, stride=self.hop_size,
            padding=self.padding, output_padding=self.output_padding)
        B, T = spec.size(0), spec.size(-1)
        window_square = self.window_square.expand(B, -1, T)
        L = self.hop_size*T + (self.N-self.hop_size) - 2*self.padding + self.output_padding
        window_square_inverse = F.fold(
            window_square,
            output_size = (1, L),
            kernel_size = (1, self.N),
            stride = (1, self.hop_size),
            padding = (0, self.padding)
        ).squeeze(2)

        # NOLA(Nonzero Overlap-add) constraint
        assert torch.all(torch.ne(window_square_inverse, 0.0))
        return wav / window_square_inverse


class MDCT(torch.jit.ScriptModule):
    '''Modified Discrete Cosine Transform
    forward(x, inverse=False):
        y: [B, 1, N * T] -> pad N to left & right each.
        output: [B, N, T + 1]
    forward(x, inverse=True):
        y: [B, N, T + 1]
        output: [B, 1, N * T]'''

    __constants__ = ["N", "filter", "normalize"]

    def __init__(self, N: int, normalize: bool = True, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.normalize = normalize

        k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1, 1)
        n = torch.arange(2*N, dtype=torch.float32, device=device).view(1, 1, 2*N)
        mdct_filter = torch.cos(math.pi/N*(n+0.5+N/2)*(k+0.5))
        if normalize:
            mdct_filter /= math.sqrt(N)
        mdct_filter = mdct_filter.to(device=device, dtype=dtype)
        self.register_buffer("filter", mdct_filter)
        self.filter: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.filter, bias=None, stride=self.N, padding=self.N)
    
    @torch.jit.export
    def inverse(self, x: Tensor) -> Tensor:
        if self.normalize:
            mdct_filter = self.filter
        else:
            mdct_filter = self.filter / self.N
        return F.conv_transpose1d(x, mdct_filter, bias=None, stride=self.N, padding=self.N)


class STFT(nn.Module):
    '''Short Time Fourier Transform
    if center == False:
        forward(x):
            x: [B, T_wav] or [B, 1, T_wav]
            output: [B, n_fft//2+1, T_wav//hop_size, 2]   (magnitude = False)
            output: [B, n_fft//2+1, T_wav//hop_size]      (magnitude = True)
        inverse(x):
            x: [B,  n_fft//2+1, T_wav//hop_size, 2]
            output: [B, T_wav]
    if center == True:
        forward(x):
            x: [B, T_wav] or [B, 1, T_wav]
            output: [B, n_fft//2+1, T_wav//hop_size+1, 2] (magnitude = False)
            output: [B, n_fft//2+1, T_wav//hop_size+1]    (magnitude = True)
        inverse(x):
            x: [B,  n_fft//2, T_wav//hop_size, 2]
            output: [B, T_wav]
    '''

    __constants__ = ["n_fft", "hop_size", "normalize", "pad_mode"]
    __annotations__ = {'window': Optional[Tensor]}

    def __init__(self, n_fft: int, hop_size: int, win_size: Optional[int] = None,
                 center: bool = True, magnitude: bool = True,
                 win_type: Optional[str] = "hann",
                 window: Optional[Tensor] = None, normalized: bool = False,
                 pad_mode: str = "reflect",
                 backend: str = "torch", device=None, dtype=None):
        super().__init__()
        assert backend in ["torch", "custom"], backend
        
        self.backend = backend
        self.normalized = normalized
        self.center = center
        self.magnitude = magnitude
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = 0 if center else (n_fft + 1 - hop_size) // 2
        self.clip = (hop_size % 2 == 1)
        self.pad_mode = pad_mode
        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
        elif win_type is None:
            window = torch.ones(win_size, device=device, dtype=dtype)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size,
                device=device, dtype=dtype)
        self.register_buffer("window", window)
        self.window: Tensor
        self.win_size = win_size
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"
        
        if backend == "custom":
            raise NotImplementedError("currently backend==custom is not implemented.")

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T_wav] or [B, 1, T_wav]
        # output: [B, n_fft//2+1, T_spec(, 2)]
        if x.dim() == 3:  # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
        if self.padding > 0:
            x = F.pad(x.unsqueeze(0), (self.padding, self.padding), mode=self.pad_mode).squeeze(0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                window=self.window, center=self.center, pad_mode=self.pad_mode,
                normalized=False, onesided=True, return_complex=False)
        
        if self.magnitude:
            spec = torch.linalg.norm(spec, dim=-1)
        
        if self.clip:
            spec = spec[:, :, :-1]

        return spec

    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, n_fft//2+1, T_spec, 2]
        # output: [B, T_wav]
        if not self.center:
            raise NotImplementedError("center=False is currently not implemented. "
                "Please set center=True")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                win_length=self.win_size, center=self.center, normalized=self.normalized,
                window=self.window, onesided=True, return_complex=False)

        return wav


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    def __init__(self, subbands=4, taps=62, cutoff_freq=0.142, beta=9.0):
        super().__init__()
        h_proto = torch.from_numpy(design_prototype_filter(taps, cutoff_freq, beta)).to(dtype=torch.float32).unsqueeze(0)
        k = torch.arange(subbands, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(taps + 1, dtype=torch.float32).unsqueeze(0)
        pqmf_filter = 2 * h_proto * torch.cos(
            (2*k + 1) * np.pi / (2 * subbands) * (n - taps / 2) + (-1)**k * np.pi / 4
        ).unsqueeze(1) * subbands ** 0.5
        self.taps = taps
        self.subbands = subbands
        self.register_buffer("pqmf_filter", pqmf_filter)
        self.pqmf_filter: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        return self.analysis(x)
    
    def analysis(self, x: Tensor) -> Tensor:
        if x.dim() == 2:  # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
        x = F.conv1d(x, self.pqmf_filter, None, stride=self.subbands, padding=self.taps//2)
        return x
    
    def synthesis(self, x: Tensor) -> Tensor:
        padding = self.taps // 2
        w = self.pqmf_filter
        x = F.conv_transpose1d(x, w, None, stride=self.subbands, padding=padding,
                               output_padding=self.subbands-1)
        return x
