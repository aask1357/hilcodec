# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from . import (
    SConv1d,
    SConvTranspose1d,
    SLSTM,
    CausalSTFT
)

from functional import STFT


def dws_conv_block(
    act: nn.Module, activation_params: dict, in_chs: int, out_chs: int, kernel_size: int,
    stride: int = 1, dilation: int = 1, norm: str = "weight_norm", norm_params: dict = {},
    causal: bool = False, pad_mode: str = 'reflect', act_all: bool = False,
    transposed: bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
) -> tp.List[nn.Module]:
    block = [
        act(**activation_params),
        SConv1d(in_chs, out_chs, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                bias=bias if act_all else False,
                nonlinearity='relu'),
    ]
    if act_all:
        block.append(act(**activation_params))
    
    Conv = SConvTranspose1d if transposed else SConv1d
    if groups == -1:
        groups = out_chs // expansion
    block.append(
        Conv(
            out_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation,
            groups=groups, norm=norm, norm_kwargs=norm_params, causal=causal,
            pad_mode=pad_mode, bias=bias,
            nonlinearity='relu' if act_all else 'linear'
        )
    )
    return block


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
        dws (bool) : Depth-Wise Separable Conv
        act_all (bool) : (Used only when dws=True) Whether to insert activation before SepConv & DWConv or before a SepConv only.
    """
    def __init__(
        self, dim: int, kernel_size: int = 3,
        dilations: tp.List[int] = [1, 1], activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},  norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
        pad_mode: str = 'reflect', skip: str = '1x1',
        act_all : bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
        res_scale: tp.Optional[float] = None, idx: int = 0, zero_init: bool = True,
    ):
        super().__init__()
        act = getattr(nn, activation)
        block = []
        inplace_act_params = activation_params.copy()
        inplace_act_params["inplace"] = True
        self.pre_scale = (1 + idx * res_scale**2)**-0.5 if res_scale is not None else None
        for i, dilation in enumerate(dilations):
            _activation_params = activation_params if i == 0 else inplace_act_params
            block += dws_conv_block(
                act,
                inplace_act_params, # _activation_params,
                dim,
                dim,
                kernel_size,
                dilation=dilation,
                norm=norm,
                norm_params=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                bias=bias,
            )
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        
        self.scale = None
        self.exp_scale = False
        if skip == "identity":
            self.shortcut = nn.Identity()
        elif skip == "1x1":
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    bias=bias)
        elif skip == "scale":
            self.scale = nn.Parameter(torch.ones(1, 1, 1))
        elif skip == "exp_scale":
            self.scale = nn.Parameter(torch.zeros(1, 1, 1))
            self.exp_scale = True
        elif skip == "channelwise_scale":
            self.scale = nn.Parameter(torch.ones(1, dim, 1))
        
        self.res_scale = res_scale
        if zero_init:
            self.res_scale_param = nn.Parameter(torch.zeros(1))
        else:
            self.res_scale_param = None

    def forward(self, x: Tensor) -> Tensor:
        # shortcut
        if self.scale is not None:
            scale = self.scale
            if self.exp_scale:
                scale = scale.exp()
            shortcut = scale * x
        else:
            shortcut = self.shortcut(x)
        
        # block
        if self.pre_scale is not None:
            x = x * self.pre_scale
        y: Tensor = self.block(x)
        
        # residual connection
        scale = 1.0 if self.res_scale is None else self.res_scale
        if self.res_scale_param is not None:
            scale = scale * self.res_scale_param
        return y.mul_(scale).add_(shortcut)


class L2Norm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-12, inout_norm: bool = True):
        super().__init__()
        self.scale = channels ** 0.5
        self.eps = eps
        self.inout_norm = inout_norm
    
    def forward(self, x: Tensor) -> Tensor:
        y = nn.functional.normalize(x, p=2.0, dim=1, eps=self.eps)
        if self.inout_norm:
            return y.mul_(self.scale)
        return y


class Scale(nn.Module):
    def __init__(self, dim: int, value: float = 1.0,
                 learnable: bool = True, inplace: bool = False):
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.ones(1, dim, 1) * value)
        else:
            self.scale = value
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.mul_(self.scale)
        return self.scale * x


class SpecBlock(nn.Module):
    def __init__(
        self, spec: str, spec_layer: str, spec_compression: str,
        n_fft: int, channels: int, stride: int, norm: str, norm_params: tp.Dict[str, tp.Any],
        bias: bool, pad_mode: str, learnable: bool, causal: bool = True,
        mean: float = 0.0, std: float = 1.0, res_scale: tp.Optional[float] = 1.0,
        zero_init: bool = True, inout_norm: bool = True,
    ) -> None:
        super().__init__()
        self.clip = False
        self.learnable = learnable
        if spec == "stft":
            if causal:
                self.spec = CausalSTFT(n_fft=n_fft, hop_size=stride,
                                       pad_mode=pad_mode, learnable=learnable)
            else:
                self.spec = STFT(n_fft=n_fft, hop_size=stride,
                                 center=False, magnitude=True)
        elif spec == "":
            self.spec = None
            return
        else:
            raise ValueError(f"Unknown spec: {spec}")
        
        if spec_compression == "log":
            self.compression = "log"
        elif spec_compression == "":
            self.compression = ""
        else:
            self.compression = float(spec_compression)

        self.inout_norm = inout_norm
        self.mean, self.std = mean, std
        self.scale = res_scale
        self.scale_param = None
        if spec_layer == "1x1":
            self.layer = SConv1d(n_fft//2+1, channels, 1, norm=norm, norm_kwargs=norm_params,
                                 bias=bias, pad_mode=pad_mode)
        elif spec_layer == "1x1_zero":
            self.layer = SConv1d(n_fft//2+1, channels, 1, norm=norm, norm_kwargs=norm_params,
                                 bias=bias, pad_mode=pad_mode)
            if zero_init:
                self.scale_param = nn.Parameter(torch.zeros(1))
        else:
            raise RuntimeError(spec_layer)

    def forward(self, x: Tensor, wav: Tensor) -> Tensor:
        if self.spec is None:
            return x
        # spectrogram
        y: Tensor = self.spec(wav)

        # compression
        if self.compression == "log":
            y = y.clamp_min(1e-5).log_()
        elif self.compression == "":
            pass
        else:
            y = y.sign() * y.abs().pow(self.compression)
        
        # normalize
        if self.inout_norm:
            y.sub_(self.mean).div_(self.std)
        
        # layer
        y = self.layer(y)
        
        # scaling
        scale = 1.0 if self.scale is None else self.scale
        if self.scale_param is not None:
            scale = self.scale_param * scale
        x.add_(y.mul_(scale))
        return x


class SEANetEncoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_fft_base: int = 64,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, skip: str = '1x1',
        causal: bool = False, pad_mode: str = 'reflect',
        act_all: bool = False, expansion: int = 1, groups: int = -1,
        l2norm: bool = False, bias: bool = True, spec: str = "stft",
        spec_layer: str = "1x1", spec_compression: str = "",
        spec_learnable: bool = False,
        res_scale: tp.Optional[float] = None,
        wav_std: float = 0.1122080159,
        spec_means: tp.List[float] = [-4.554, -4.315, -4.021, -3.726, -3.477],
        spec_stds: tp.List[float] = [2.830, 2.837, 2.817, 2.796, 2.871],
        zero_init: bool = True,
        inout_norm: bool = True,
    ):
        super().__init__()
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        
        act = getattr(nn, activation)
        mult = 1
        self.conv_pre = nn.Sequential(
            Scale(1, value=1/wav_std, learnable=False, inplace=False) if inout_norm else nn.Identity(),
            SConv1d(
                channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode, bias=bias
            )
        )
        
        self.blocks = nn.ModuleList()
        self.spec_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        stride = 1
        for block_idx, ratio in enumerate(self.ratios):
            # Add residual layers
            block = []
            for j in range(1, n_residual_layers + 1):
                idx = j - 1 if spec == "" else j
                block += [
                    SEANetResnetBlock(mult * n_filters, kernel_size=residual_kernel_size,
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode,
                                      skip=skip, act_all=act_all,
                                      expansion=expansion, groups=groups, bias=bias,
                                      res_scale=res_scale, idx=idx,
                                      zero_init=zero_init,)
                ]
            self.blocks.append(nn.Sequential(*block))
            
            # add spectrogram layer
            spec_block = SpecBlock(
                spec, spec_layer, spec_compression, mult*n_fft_base, mult*n_filters, stride,
                norm, norm_params, bias=False, pad_mode=pad_mode,
                learnable=spec_learnable, causal=causal,
                mean=spec_means[block_idx], std=spec_stds[block_idx], res_scale=res_scale,
                zero_init=zero_init, inout_norm=inout_norm,
            )
            self.spec_blocks.append(spec_block)
            stride *= ratio

            # Add downsampling layers
            if res_scale is None:
                scale_layer = nn.Identity()
            else:
                scale_layer = Scale(
                    1, value=(1+n_residual_layers*res_scale**2)**-0.5,
                    learnable=False, inplace=True
                )
            downsample = nn.Sequential(
                scale_layer,
                act(inplace=True, **activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2, 1,
                        norm=norm, norm_kwargs=norm_params, bias=False,
                        nonlinearity='relu'),
                SConv1d(mult * n_filters * 2, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters*2,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode, bias=bias),
            )
            self.downsample.append(downsample)
            mult *= 2

        self.spec_post = SpecBlock(
            spec, spec_layer, spec_compression, mult*n_fft_base, mult*n_filters,
            stride, norm, norm_params, bias=False, pad_mode=pad_mode,
            learnable=spec_learnable, causal=causal,
            mean=spec_means[-1], std=spec_stds[-1], res_scale=res_scale, zero_init=zero_init,
            inout_norm=inout_norm,
        )
        self.conv_post = nn.Sequential(
            act(inplace=False, **activation_params),
            SConv1d(mult * n_filters, mult * n_filters, last_kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode,
                    bias=False, nonlinearity='relu'),
            SConv1d(mult * n_filters, dimension, 1, norm=norm,
                    norm_kwargs=norm_params, bias=bias),
            L2Norm(dimension, inout_norm=inout_norm) if l2norm else nn.Identity(),
        )
        if l2norm:
            # If the input audio has silent frames,
            # the encoder output will be near-zero vector sequence (right after initialization).
            # With l2norm, this will result in big scaling at the forward & backward,
            # meaning huge gradient will be encountered.
            # To avoid this, we initialize the last conv layer with
            # big non-zero bias.
            self.conv_post[-2].conv.conv.bias.data.normal_()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, T]
        wav = x
        x = self.conv_pre(x)
        for block, spec_block, downsample in zip(self.blocks, self.spec_blocks, self.downsample):
            x = spec_block(x, wav)
            x = block(x)
            x = downsample(x)
        x = self.spec_post(x, wav)
        x = self.conv_post(x)
        return x


class SEANetDecoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, skip: str = '1x1',
        causal: bool = False, pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        act_all: bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
        res_scale: tp.Optional[float] = None,
        wav_std: float = 0.1122080159,
        zero_init: bool = True,
        inout_norm: bool = True,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = []
        model += [
            SConv1d(dimension, mult * n_filters, 1, norm=norm, norm_kwargs=norm_params, bias=False),
            SConv1d(mult * n_filters, mult * n_filters, kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias)
        ]

        j = 0
        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            if i > 0:
                if res_scale is None:
                    scale_layer = nn.Identity()
                else:
                    scale_layer = Scale(
                        1, value=(1+n_residual_layers*res_scale**2)**-0.5,
                        learnable=False, inplace=True
                    )
            else:
                scale_layer = nn.Identity()
            model += [
                scale_layer,
                act(inplace=True, **activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters,
                                 kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio, bias=False,
                                 nonlinearity='relu'),
                SConv1d(mult * n_filters, mult * n_filters // 2, 1,
                        norm=norm, norm_kwargs=norm_params, bias=bias)
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_size=residual_kernel_size,
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, skip=skip,
                                      act_all=act_all, expansion=expansion, groups=groups,
                                      bias=bias, res_scale=res_scale, idx=j, zero_init=zero_init,)]
            mult //= 2

        # Add final layers
        if res_scale is None:
            scale_layer = nn.Identity()
        else:
            scale_layer = Scale(
                1, value=(1+n_residual_layers*res_scale**2)**-0.5, learnable=False, inplace=True
            )
        model += [
            scale_layer,
            act(inplace=True, **activation_params),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias, nonlinearity='relu'),
            Scale(1, value=wav_std, learnable=False, inplace=True) if inout_norm else nn.Identity(),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y
