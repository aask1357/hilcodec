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
                bias=bias if act_all else False),
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
            pad_mode=pad_mode, bias=bias
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
        pad_mode: str = 'reflect', compress: int = 2, skip: str = '1x1',
        act_all : bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
        res_scale: tp.Optional[float] = None,
    ):
        super().__init__()
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        inplace_act_params = activation_params.copy()
        inplace_act_params["inplace"] = True
        for i, dilation in enumerate(dilations):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(dilations) - 1 else hidden
            _activation_params = activation_params if i == 0 else inplace_act_params
            block += dws_conv_block(
                act,
                _activation_params,
                in_chs,
                out_chs,
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
        if skip == "identity":
            self.shortcut = nn.Identity()
        elif skip == "1x1":
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    bias=bias)
        elif skip == "scale":
            self.scale = nn.Parameter(torch.ones(1, 1, 1))
        elif skip == "channelwise_scale":
            self.scale = nn.Parameter(torch.ones(1, dim, 1))
        
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.block(x)
        if self.res_scale is not None:
            y.mul_(self.res_scale)
        if self.scale is not None:
            return y.addcmul_(self.scale, x)    # self.block(x) + self.scale * x
        else:
            return y.add_(self.shortcut(x))        # self.block(x) + x


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(x, p=2.0, dim=1, eps=self.eps)


class Scale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x


class SpecBlock(nn.Module):
    def __init__(self, spec: str, spec_layer: str, spec_compression: str,
                 channels: int, stride: int, norm: str, norm_params: tp.Dict[str, tp.Any],
                 bias: bool, pad_mode: str, leanable: bool, causal: bool = True) -> None:
        super().__init__()
        self.clip = False
        if spec == "stft":
            if causal:
                self.spec = CausalSTFT(n_fft=2*channels, hop_size=stride,
                                       pad_mode=pad_mode, learnable=leanable)
            else:
                self.spec = STFT(n_fft=2*channels, hop_size=stride,
                                 center=False, magnitude=True)
            self.clip = True
            in_channels = channels + 1
        elif spec == "":
            self.spec = None
            return
        else:
            raise ValueError(f"Unknown spec: {spec}")
        
        if spec_compression == "log":
            self.compression = "log"
        elif spec_compression == "abs_log":
            self.compression = "abs_log"
        elif spec_compression == "":
            self.compression = ""
        else:
            self.compression = float(spec_compression)

        self.div2 = False
        if spec_layer == "1x1":
            self.layer = SConv1d(in_channels, channels, 1, norm=norm, norm_kwargs=norm_params,
                                 bias=bias, pad_mode=pad_mode)
            self.clip = False
        elif spec_layer == "1x1_div2":
            self.layer = SConv1d(in_channels, channels, 1, norm=norm, norm_kwargs=norm_params,
                                 bias=bias, pad_mode=pad_mode)
            self.clip = False
            self.div2 = True
        elif spec_layer == "1x1_zero":
            self.layer = SConv1d(in_channels, channels, 1, norm=norm, norm_kwargs=norm_params,
                                 bias=bias, pad_mode=pad_mode)
            self.layer.conv.conv.weight_g.data.zero_()
            self.clip = False
        elif spec_layer == "scale":
            self.layer = Scale(1)
        elif spec_layer == "channelwise_scale":
            self.layer = Scale(channels)

    def forward(self, x: Tensor, wav: Tensor) -> Tensor:
        if self.spec is None:
            return x
        # spectrogram
        y: Tensor = self.spec(wav)
        if self.clip:
            y = y[:, :-1, :]

        # compression
        if self.compression == "log":
            y = y.clamp_min(1e-5).log()
        elif self.compression == "abs_log":
            y = y.abs().clamp_min(1e-5).log()
        elif self.compression == "":
            pass
        else:
            y = y.sign() * y.abs().pow(self.compression)
        
        # layer
        x = x + self.layer(y)
        if self.div2:
            x /= 2.0
        return x


class SEANetEncoder(nn.Module):
    """SEANet encoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
                 n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
                 kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
                 dilation_base: int = 2, skip: str = '1x1', compress: int = 2, lstm: int = 2,
                 causal: bool = False, pad_mode: str = 'reflect',
                 act_all: bool = False, expansion: int = 1, groups: int = -1,
                 l2norm: bool = False, bias: bool = True, spec: str = "stft",
                 spec_layer: str = "1x1", spec_compression: str = "",
                 spec_learnable: bool = False,
                 res_scale: tp.Optional[float] = None, mu: bool = False,):
        super().__init__()
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.mu = mu
        self._denominator_mu = math.log(1.0 + 255.0)
        
        act = getattr(nn, activation)
        mult = 1
        channels = channels * 2 if mu else channels
        self.conv_pre = SConv1d(
            channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
            causal=causal, pad_mode=pad_mode, bias=bias)
        
        self.blocks = nn.ModuleList()
        self.spec_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        stride = 1
        for ratio in self.ratios:
            # Add residual layers
            block = []
            for j in range(1, n_residual_layers + 1):
                block += [
                    SEANetResnetBlock(mult * n_filters, kernel_size=residual_kernel_size,
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress,
                                      skip=skip, act_all=act_all,
                                      expansion=expansion, groups=groups, bias=bias,
                                      res_scale=res_scale)
                ]
            self.blocks.append(nn.Sequential(*block))
            
            # add spectrogram layer
            spec_block = SpecBlock(spec, spec_layer, spec_compression, mult * n_filters, stride,
                                   norm, norm_params, bias=False, pad_mode=pad_mode,
                                   leanable=spec_learnable, causal=causal)
            self.spec_blocks.append(spec_block)
            stride *= ratio

            # Add downsampling layers
            downsample = nn.Sequential(
                act(inplace=True, **activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2, 1,
                        norm=norm, norm_kwargs=norm_params, bias=False),
                SConv1d(mult * n_filters * 2, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters*2,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode, bias=bias),
            )
            self.downsample.append(downsample)
            mult *= 2

        self.spec_post = SpecBlock(spec, spec_layer, spec_compression, mult * n_filters,
                                   stride, norm, norm_params, bias=False, pad_mode=pad_mode,
                                   leanable=spec_learnable, causal=causal)
        self.conv_post = nn.Sequential(
            act(inplace=True, **activation_params),
            SConv1d(mult * n_filters, mult * n_filters, last_kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode,
                    bias=False),
            SConv1d(mult * n_filters, dimension, 1, norm=norm,
                    norm_kwargs=norm_params, bias=bias),
            L2Norm() if l2norm else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, T]
        wav = x
        if self.mu:
            x_encoded = x.sign() * (1.0 + 255.0*x.abs()).log() / self._denominator_mu
            x = torch.cat([x, x_encoded], dim=1)
        x = self.conv_pre(x)
        for block, spec_block, downsample in zip(self.blocks, self.spec_blocks, self.downsample):
            x = spec_block(x, wav)
            x = block(x)
            x = downsample(x)
        x = self.spec_post(x, wav)
        x = self.conv_post(x)
        return x


class SEANetDecoder(nn.Module):
    """SEANet decoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, skip: str = '1x1', compress: int = 2, lstm: int = 2,
        causal: bool = False, pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        act_all: bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
        res_scale: tp.Optional[float] = None,
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
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, 1, norm=norm, norm_kwargs=norm_params, bias=False),
            SConv1d(mult * n_filters, mult * n_filters, kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                act(inplace=True, **activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters,
                                 kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio, bias=False),
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
                                      pad_mode=pad_mode, compress=compress, skip=skip,
                                      act_all=act_all, expansion=expansion, groups=groups,
                                      bias=bias, res_scale=res_scale)]
            mult //= 2

        # Add final layers
        model += [
            act(inplace=True, **activation_params),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias)
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
