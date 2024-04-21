# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MS-STFT discriminator, provided here for reference."""

import typing as tp

import torchaudio
import torch
from torch import Tensor
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.cuda import amp

from functional import PQMF

from .modules import NormConv2d
from .avocodo.CoMBD import CoMBD
from .avocodo.SBD import SBD


LRELU_SLOPE = 0.1


DiscOutput = tp.Tuple[tp.List[Tensor], tp.List[Tensor]]
FinalDiscOutput = tp.Tuple[tp.Dict[str, tp.List[Tensor]], tp.Dict[str, tp.List[Tensor]]]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


def get_2d_padding(
    kernel_size: tp.Tuple[int, int],
    dilation: tp.Tuple[int, int] = (1, 1)
) -> tp.Tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2
    )


class STFTDiscriminator(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 max_filters: int = 1024, filters_scale: int = 1,
                 kernel_size: tp.Tuple[int, int] = (3, 9),
                 dilations: tp.List[int] = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True,
                 norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2, 'inplace': True},
                 magnitude: bool = False, conv_pre_no_weight_norm: bool = False,
                 log_magnitude: bool = False, eps: float = 1e-5):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.magnitude = magnitude
        self.log_magnitude = log_magnitude
        self.eps = eps
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window_fn=torch.hann_window, normalized=self.normalized, center=False,
            pad_mode=None, power=None)
        spec_channels = self.in_channels if magnitude else 2 * self.in_channels
        self.convs = nn.ModuleList()
        _norm = 'none' if conv_pre_no_weight_norm else norm
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size,
                       padding=get_2d_padding(kernel_size), norm=_norm)
        )
        in_chs = min(self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** i) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size,
                                         stride=stride, dilation=(dilation, 1),
                                         padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** len(dilations)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs,
                                     kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        fmap = []
        z = self.spec_transform(x)  # [B, 1, Freq, Frames] (complex)
        if self.magnitude:
            z = z.abs()
            z = z.transpose(2, 3)       # [B, 1, Time, Freq]
            if getattr(self, "log_magnitude", False):
                z = torch.log(z + self.eps)
        else:
            z = torch.view_as_real(z)   # [B, 1, Freq, Time, 2]
            z = z.squeeze(1).permute(0, 3, 2, 1).contiguous()    # [B, 2, Time, Freq]
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiSTFTDiscriminator(nn.Module):
    """Multi STFT (M-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int] = [256, 512, 128, 64, 32],
                 win_lengths: tp.List[int] = [1024, 2048, 512, 256, 128],
                 magnitude: bool = False, **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i],
                              magnitude=magnitude, **kwargs)
            for i in range(len(n_ffts))
        ])

    def forward(self, x: Tensor) -> DiscOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.extend(fmap)
        return logits, fmaps


class FilterBankDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        taps: int = 0,
        beta: float = 0.0,
        cutoff_freq: float = 0.0,
        kernel_sizes: tp.List[int] = [5, 5, 5, 5, 5],
        strides: tp.List[int] = [3, 3, 3, 3, 3, 1],
        channels: tp.List[int] = [32, 128, 256, 512, 1024, 1024],
        norm: str = "weight_norm",
    ):
        super().__init__()
        self.period = period
        if period == 1:
            self.pqmf = nn.Identity()
        else:
            assert taps > 0 and beta > 0.0 and cutoff_freq > 0.0
            self.pqmf = PQMF(subbands=period, taps=taps, beta=beta, cutoff_freq=cutoff_freq)
        
        if norm == "weight_norm":
            norm_f = weight_norm
        elif norm == "spectral_norm":
            norm_f = spectral_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")
        c_in = 1
        
        self.convs = nn.ModuleList([])
        for (ch, s, k) in zip(channels, strides, kernel_sizes):
            kernel_size = (1, k)
            padding = (0, get_padding(k))
            conv = norm_f(nn.Conv2d(c_in, ch, kernel_size, (1, s), padding=padding))
            self.convs.append(conv)
            c_in = ch
        
        self.conv_post = norm_f(nn.Conv2d(c_in, 1, (1, 3), 1, padding=(0, 1)))

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # x: [Batch, 1, Time]
        fmap = []

        x = self.pqmf(x).unsqueeze(1)   # [B, 1, Period, T//Period]

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiFilterBankDiscriminator(nn.Module):
    def __init__(
        self,
        periods: tp.List[int] = [1, 2, 3, 5, 7, 11],
        taps: int = 256,
        beta: float = 8.0,
        cutoff_freqs: tp.List[float] = [0, 0.253881, 0.170546, 0.103881, 0.075310, 0.049338],
        kernel_sizes: tp.List[int] = [5, 5, 5, 5, 5],
        strides: tp.List[int] = [3, 3, 3, 3, 1],
        channels: tp.List[int] = [32, 128, 512, 1024, 1024],
        norm: str = "weight_norm",
    ):
        assert len(strides) == len(channels) == len(kernel_sizes)
        super().__init__()
        discs = [
            FilterBankDiscriminator(
                p, taps=taps, beta=beta, cutoff_freq=c, kernel_sizes=kernel_sizes,
                strides=strides, channels=channels, norm=norm
            ) for p, c in zip(periods, cutoff_freqs)
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, ys: tp.List[Tensor]) -> DiscOutput:
        y = ys[-1]
        y_ds = []
        fmaps = []
        for disc in self.discriminators:
            y_d, fmap = disc(y)
            y_ds.append(y_d)
            fmaps.extend(fmap)

        return y_ds, fmaps


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, norm: str = "weight_norm"):
        super(DiscriminatorP, self).__init__()
        self.period = period
        if norm == "weight_norm":
            norm_f = weight_norm
        elif norm == "spectral_norm":
            norm_f = spectral_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, kernel_size: int = 5, stride: int = 3,
                 norm: str = "weight_norm"):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, kernel_size, stride, norm=norm),
            DiscriminatorP(3, kernel_size, stride, norm=norm),
            DiscriminatorP(5, kernel_size, stride, norm=norm),
            DiscriminatorP(7, kernel_size, stride, norm=norm),
            DiscriminatorP(11, kernel_size, stride, norm=norm),
        ])

    def forward(self, y: Tensor) -> DiscOutput:
        y_ds = []
        fmaps = []
        for d in self.discriminators:
            y_d, fmap = d(y)
            y_ds.append(y_d)
            fmaps.extend(fmap)

        return y_ds, fmaps


class DiscriminatorS(torch.nn.Module):
    def __init__(self, norm: str = "weight_norm"):
        super(DiscriminatorS, self).__init__()
        if norm == "weight_norm":
            norm_f = weight_norm
        elif norm == "spectral_norm":
            norm_f = spectral_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, norm: tp.Optional[str] = None):
        super(MultiScaleDiscriminator, self).__init__()
        if norm is None:    # default
            norms = ["spectral_norm", "weight_norm", "weight_norm"]
        else:
            norms = [norm for _ in range(3)]
        self.discriminators = nn.ModuleList([
            DiscriminatorS(norms[0]),
            DiscriminatorS(norms[1]),
            DiscriminatorS(norms[2]),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y: Tensor) -> DiscOutput:
        y_ds = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
            y_d, fmap = d(y)
            y_ds.append(y_d)
            fmaps.extend(fmap)

        return y_ds, fmaps


class Discriminators(nn.Module):
    def __init__(
        self,
        mfbd_kwargs: tp.Dict[str, tp.Any] = {},
        mpd_kwargs: tp.Dict[str, tp.Any] = {},
        msd_kwargs: tp.Dict[str, tp.Any] = {},
        mstftd_kwargs: tp.Dict[str, tp.Any] = {},
        combd_kwargs: tp.Dict[str, tp.Any] = {},
        sbd_kwargs: tp.Dict[str, tp.Any] = {},
    ) -> None:
        super().__init__()
        self.discs = nn.ModuleDict()
        if mfbd_kwargs.pop("use", False):
            self.discs["mfbd"] = MultiFilterBankDiscriminator(**mfbd_kwargs)
        if mpd_kwargs.pop("use", False):
            self.discs["mpd"] = MultiPeriodDiscriminator(**mpd_kwargs)
        if msd_kwargs.pop("use", False):
            self.discs["msd"] = MultiScaleDiscriminator(**msd_kwargs)
        if mstftd_kwargs.pop("use", False):
            self.discs["mstftd"] = MultiSTFTDiscriminator(**mstftd_kwargs)
        if combd_kwargs.pop("use", False):
            self.discs["combd"] = CoMBD(**combd_kwargs)
        if sbd_kwargs.pop("use", False):
            self.discs["sbd"] = SBD(**sbd_kwargs)
    
    @amp.autocast(enabled=False)
    def forward(self, y) -> FinalDiscOutput:
        y_ds, fmaps = {}, {}
        for name, disc in self.discs.items():
            y_d, fmap = disc(y)
            y_ds[name] = y_d
            fmaps[name] = fmap
        return y_ds, fmaps
