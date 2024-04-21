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
from torch import Tensor, nn, jit
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm

from functional import STDCT
from utils import verbose

from .causal_layers import SConv1d, SConvTranspose1d, STFT, STDCT, STFTOnnx
from typing import Optional, List


class EuclideanCodebook(nn.Module):
    __constants__ = ['codebook_size', 'eps', 'ema_num_initial']
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 20,
        decay: float = 0.8,
        eps: float = 1e-7,
        ema_num_threshold: float = 0.0,
        ema_num_initial: float = 1.0,
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.eps = eps
        self.ema_num_initial = ema_num_initial

        self.register_buffer('embed', embed)
        self.register_buffer('ema_num', torch.ones(codebook_size) * ema_num_initial)
        self.embed: Tensor
        self.ema_num: Tensor

    def forward(self, x: Tensor) -> Tensor:
        # x.dtype: float16 / embed.dtype: float32 / device: cuda{rank}
        B, T, C = x.shape
        flatten = x.reshape(B * T, C)
        embed = self.embed.t()                          # [Channel, codebook_size]

        # distance: [Batch x Time, codebook_size]
        distance = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = distance.max(dim = -1).indices
        embed_ind = embed_ind.view(B, T)
        quantize = F.embedding(embed_ind, self.embed)

        return quantize


class VectorQuantize(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._codebook = EuclideanCodebook(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)   # [B, C, T] -> [B, T, C]
        quantize: Tensor = self._codebook(x)
        return quantize.transpose(1, 2)


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers: int,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantize(**kwargs) for _ in range(num_quantizers)
        ])

    def forward(self, x: Tensor, n: int) -> Tensor:
        quantized_out = torch.zeros_like(x)

        residual = x
        for layer in self.layers[:n]:
            quantized = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
        
        return quantized_out


class DWSBlock(nn.Module):
    def __init__(
        self, act, activation_params: dict, in_chs: int, out_chs: int, kernel_size: int,
        stride: int = 1, dilation: int = 1, norm: str = "weight_norm",
        act_all: bool = False, transposed: bool = False, expansion: int = 1,
        groups: int = -1, bias: bool = True,
    ):
        super().__init__()
        block = [
            act(**activation_params),
            SConv1d(in_chs, out_chs, kernel_size=1, norm=norm,
                bias=bias if act_all else False),
        ]
        if act_all:
            block.append(act(**activation_params))
        self.pointwise = nn.Sequential(*block)
        
        Conv = SConvTranspose1d if transposed else SConv1d
        if groups == -1:
            groups = out_chs // expansion
        
        self.depthwise = Conv(
            out_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation,
            groups=groups, norm=norm, bias=bias
        )
    
    def initialize_cache(self, x: Tensor) -> Tensor:
        return self.depthwise.initialize_cache(x)
    
    def forward(self, x: Tensor, cache: Tensor) -> tp.Tuple[Tensor, Tensor]:
        x = self.pointwise(x)
        x, cache = self.depthwise(x, cache)
        return x, cache


class ResBlock(nn.Module):
    def __init__(
        self, dim: int, kernel_size: int = 3,
        dilations: tp.List[int] = [1, 1], activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},  norm: str = 'weight_norm',
        compress: int = 2, skip: str = '1x1',
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
            block.append(DWSBlock(
                act,
                _activation_params,
                in_chs,
                out_chs,
                kernel_size,
                dilation=dilation,
                norm=norm,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                bias=bias,
            ))
        self.block = nn.ModuleList(block)
        self.shortcut: nn.Module
        
        self.scale = None
        if skip == "identity":
            self.shortcut = nn.Identity()
        elif skip == "1x1":
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm,
                                    bias=bias)
        elif skip == "scale":
            self.scale = nn.Parameter(torch.ones(1, 1, 1))
        elif skip == "channelwise_scale":
            self.scale = nn.Parameter(torch.ones(1, dim, 1))
        
        self.res_scale = res_scale
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache: tp.List[Tensor] = []
        for block in self.block:
            cache.append(block.initialize_cache(x))
        return cache

    def forward(
        self,
        x: Tensor,
        cache: tp.List[Tensor]
    ) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        new_cache: tp.List[Tensor] = []
        skip = x
        for idx, block in enumerate(self.block):
            x, cache_idx = block(x, cache[idx])
            new_cache.append(cache_idx)
        if self.res_scale is not None:
            x.mul_(self.res_scale)
        if self.scale is not None:
            x.addcmul_(self.scale, skip)    # self.block(x) + self.scale * x
        else:
            x.add_(self.shortcut(skip))        # self.block(x) + x
        return x, new_cache


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
                 channels: int, stride: int,
                 norm: str, bias: bool) -> None:
        super().__init__()
        if spec == "dct":
            self.spec = STDCT(N=channels, hop_size=stride)
            in_channels = channels
        elif spec == "stft":
            self.spec = STFTOnnx(n_fft=2*channels, hop_size=stride, magnitude=True)
            in_channels = channels + 1
        
        if spec_compression == "log":
            self.compression = "log"
        elif spec_compression == "abs_log":
            self.compression = "abs_log"
        elif spec_compression == "":
            self.compression = ""
        else:
            self.compression = float(spec_compression)

        if spec_layer == "1x1":
            self.layer = SConv1d(in_channels, channels, 1, norm=norm,
                                 bias=bias)
            self.clip = False
        elif spec_layer == "scale":
            self.layer = Scale(1)
        elif spec_layer == "channelwise_scale":
            self.layer = Scale(channels)

    def forward(self, x: Tensor, wav: Tensor) -> Tensor:
        # spectrogram
        y = self.spec(wav)

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
        return x + self.layer(y)


class Encoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, skip: str = '1x1', compress: int = 2,
        act_all: bool = False, expansion: int = 1, groups: int = -1,
        l2norm: bool = False, bias: bool = True, spec: str = "dct",
        spec_layer: str = "1x1", spec_compression: str = "",
        res_scale: tp.Optional[float] = None, mu: bool = False,
    ):
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
            channels, mult * n_filters, kernel_size, norm=norm, bias=bias)
        
        self.blocks = nn.ModuleList()
        self.spec_blocks = nn.ModuleList()
        self.downsample_pointwise = nn.ModuleList()
        self.downsample_depthwise = nn.ModuleList()
        stride = 1
        for ratio in self.ratios:
            # Add residual layers
            block = []
            for j in range(1, n_residual_layers + 1):
                block += [ResBlock(
                    mult * n_filters, kernel_size=residual_kernel_size,
                    dilations=[dilation_base ** j, 1], norm=norm,
                    activation=activation, activation_params=activation_params,
                    compress=compress, skip=skip, act_all=act_all,
                    expansion=expansion, groups=groups, bias=bias,
                    res_scale=res_scale
                )]
            self.blocks.append(nn.ModuleList(block))
            
            # add spectrogram layer
            spec_block = SpecBlock(spec, spec_layer, spec_compression, mult * n_filters, stride,
                                   norm, bias)
            self.spec_blocks.append(spec_block)
            stride *= ratio

            # Add downsampling layers
            downsample_pointwise = nn.Sequential(
                act(inplace=True, **activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2, 1,
                        norm=norm, bias=False),
            )
            downsample_depthwise = SConv1d(mult * n_filters * 2, mult * n_filters * 2,
                kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters*2,
                norm=norm, bias=bias)
            self.downsample_pointwise.append(downsample_pointwise)
            self.downsample_depthwise.append(downsample_depthwise)
            mult *= 2

        self.spec_post = SpecBlock(spec, spec_layer, spec_compression,
                                   mult * n_filters, stride, norm, bias)
        self.conv_post_act = act(inplace=True, **activation_params)
        self.conv_post_depthwise = SConv1d(
            mult * n_filters, mult * n_filters, last_kernel_size, groups=mult*n_filters,
            norm=norm, bias=False)
        self.conv_post_pointwise = SConv1d(
            mult * n_filters, dimension, 1, norm=norm, bias=bias)
        self.l2norm = L2Norm() if l2norm else nn.Identity()
        self.num_cache = len(self.initialize_cache(torch.zeros(1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        wav_cache = torch.zeros(x.size(0), 1, self.spec_post.spec.cache_len, device=x.device)
        cache_out.append(wav_cache)
        cache_out.append(self.conv_pre.initialize_cache(x))
        for blocks, down_depthwise in zip(self.blocks, self.downsample_depthwise):
            for block in blocks:
                c = block.initialize_cache(x)
                assert isinstance(c, list)
                assert isinstance(c[0], Tensor)
                cache_out.extend(block.initialize_cache(x))
            cache_out.append(down_depthwise.initialize_cache(x))
        cache_out.append(self.conv_post_depthwise.initialize_cache(x))
        return cache_out

    def forward(self, x: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # x: [B, 1, T]
        cache_in: tp.List[Tensor] = [*args]
        cache_out: tp.List[Tensor] = []
        pad = self.spec_post.spec.cache_len//2
        wav = torch.cat((cache_in[0], x), dim=2)
        # wav = F.pad(x, (pad, pad)) -> for comparing with Encodec wrapper model
        cache_out.append(wav[:, :, -self.spec_post.spec.cache_len:])
        x = wav[:, :, pad:-pad]
        
        # optional mu-law encoding
        if self.mu:
            x_encoded = x.sign() * (1.0 + 255.0*x.abs()).log() / self._denominator_mu
            x = torch.cat([x, x_encoded], dim=1)
        
        x, cache = self.conv_pre(x, cache_in[1])
        cache_out.append(cache)
        
        idx = 2
        for blocks, spec_block, down_pointwise, down_depthwise in zip(
            self.blocks, self.spec_blocks, self.downsample_pointwise, self.downsample_depthwise
        ):
            pad_right = spec_block.spec.cache_len//2
            pad_left = spec_block.spec.cache_len - pad_right
            _wav = wav[:, :, pad-pad_left:-pad+pad_right]
            x = spec_block(x, _wav)

            for block in blocks:
                block: ResBlock
                n = len(block.block)
                x, cache = block(x, cache_in[idx:idx + n])
                cache_out.extend(cache)
                idx += n

            x = down_pointwise(x)
            x, cache = down_depthwise(x, cache_in[idx])
            cache_out.append(cache)
            idx += 1
        x = self.spec_post(x, wav)
        x = self.conv_post_act(x)
        x, cache = self.conv_post_depthwise(x, cache_in[idx])
        cache_out.append(cache)
        x = self.conv_post_pointwise(x)
        x = self.l2norm(x)
        return x, cache_out


class Decoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, skip: str = '1x1', compress: int = 2,
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
        self.conv_pre_pointwise = SConv1d(dimension, mult * n_filters, 1, norm=norm,
                                          bias=False)
        self.conv_pre_depthwise = SConv1d(
            mult * n_filters, mult * n_filters, kernel_size, groups=mult*n_filters,
            norm=norm, bias=bias)

        self.blocks = nn.ModuleList()
        self.upsample_act = nn.ModuleList()
        self.upsample_depthwise = nn.ModuleList()
        self.upsample_pointwise = nn.ModuleList()
        # Upsample to raw audio scale
        for ratio in self.ratios:
            # Add upsampling layers
            self.upsample_act.append(act(inplace=True, **activation_params))
            upsample_depthwise = SConvTranspose1d(mult * n_filters, mult * n_filters,
                                 kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters,
                                 norm=norm, bias=False)
            upsample_pointwise = SConv1d(
                mult * n_filters, mult * n_filters // 2, 1,
                norm=norm, bias=bias)
            self.upsample_depthwise.append(upsample_depthwise)
            self.upsample_pointwise.append(upsample_pointwise)
            
            # Add residual layers
            blocks = []
            for j in range(n_residual_layers):
                blocks += [ResBlock(
                    mult * n_filters // 2, kernel_size=residual_kernel_size,
                    dilations=[dilation_base ** j, 1],
                    activation=activation, activation_params=activation_params,
                    norm=norm, compress=compress, skip=skip,
                    act_all=act_all, expansion=expansion, groups=groups,
                    bias=bias, res_scale=res_scale
                )]
            self.blocks.append(nn.ModuleList(blocks))
            mult //= 2

        # Add final layers
        self.conv_post_act = act(inplace=True, **activation_params)
        self.conv_post = SConv1d(
            n_filters, channels, last_kernel_size, norm=norm,
            bias=bias)
        if final_activation is not None:
            final_activation_params = final_activation_params or {}
            self.final_act = getattr(nn, final_activation)(**final_activation_params) 
        else:
            self.final_act = nn.Identity()
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        cache_out.append(self.conv_pre_depthwise.initialize_cache(x))
        for blocks, up_depthwise in zip(self.blocks, self.upsample_depthwise):
            cache_out.append(up_depthwise.initialize_cache(x))
            for block in blocks:
                cache_out.extend(block.initialize_cache(x))
        cache_out.append(self.conv_post.initialize_cache(x))
        return cache_out

    def forward(self, x: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        cache_in: tp.List[Tensor] = [*args]
        cache_out: tp.List[Tensor] = []
        x = self.conv_pre_pointwise(x)
        x, cache = self.conv_pre_depthwise(x, cache_in[0])
        cache_out.append(cache)
        
        idx = 1
        for blocks, up_act, up_depthwise, up_pointwise in zip(
            self.blocks, self.upsample_act, self.upsample_depthwise, self.upsample_pointwise
        ):
            x = up_act(x)
            x, cache = up_depthwise(x, cache_in[idx])
            x = up_pointwise(x)
            cache_out.append(cache)
            idx += 1
            for block in blocks:
                block: ResBlock
                n = len(block.block)
                x, cache = block(x, cache_in[idx:idx + n])
                cache_out.extend(cache)
                idx += n
        x = self.conv_post_act(x)
        x, cache = self.conv_post(x, cache_in[idx])
        cache_out.append(cache)
        x = self.final_act(x)
        return x, cache_out


class Encodec(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        channels_audio: int = 1,
        channels_enc: int = 32,
        channels_dec: int = 32,
        n_residual_enc: int = 1,
        n_residual_dec: int = 1,
        res_scale_enc: tp.Optional[float] = None,
        res_scale_dec: tp.Optional[float] = None,
        strides: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_kwargs: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        kernel_size: int = 9,
        last_kernel_size: int = 9,
        residual_kernel_size: int = 9,
        dilation_base: int = 1,
        skip: str = "identity",
        compress: int = 1,
        lstm: int = 0,
        final_activation: tp.Optional[str] = "Tanh",
        use_vq: bool = True,    # deprecated
        vq: str = "ResidualVQ", # "" / "ResidualVQ" / "ResidualGainShapeVQ" / "ResidualGainResidualShapeVQ"
        vq_kwargs: tp.Dict[str, tp.Any] = {},
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        encoder_l2norm: bool = True,
        bias: bool = True,
        spec: str = "dct",          # dct or stft
        spec_layer: str = "1x1",    # 1x1 or scale or channelwise_scale
        spec_compression: str = "0.1", # "" or "log" or float(0~1)
        mu: bool = False,
    ):
        assert lstm == 0
        assert spec in ["dct", "stft", ""]
        assert spec_layer in ["1x1", "scale", "channelwise_scale"]
        assert skip in ["1x1", "scale", "channelwise_scale", "identity"]
        if expansion != 1 and groups != -1:
            raise RuntimeError(
                f"Both expansion({expansion}) and groups({groups}) are set. "
                f"Either set expansion=1 or set groups=-1"
            )
        if encoder_l2norm and vq != "ResidualVQ" and verbose():
            print(f"Warning: encoder_l2norm is used with vq {vq}")
        
        super().__init__()
        self.norm = norm
        channels_vq = vq_kwargs['dim']
        self.encoder = Encoder(channels_audio, channels_vq, channels_enc,
            n_residual_enc, strides, activation, activation_kwargs,
            norm, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, skip, compress,
            act_all=act_all, expansion=expansion, groups=groups, l2norm=encoder_l2norm,
            bias=bias, spec=spec, spec_layer=spec_layer, spec_compression=spec_compression,
            res_scale=res_scale_enc, mu=mu,)
        self.decoder = Decoder(channels_audio, channels_vq, channels_dec,
            n_residual_dec, strides, activation, activation_kwargs,
            norm, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, skip, compress,
            final_activation=final_activation,
            act_all=act_all, expansion=expansion, groups=groups, bias=bias,
            res_scale=res_scale_dec,)
        self.quantizer = ResidualVQ(**vq_kwargs)

        self.sample_rate = sample_rate
        self.channels = channels_audio
    
    def initialize_cache(self, x: Tensor) -> tp.Tuple[tp.List[Tensor], tp.List[Tensor]]:
        return self.encoder.initialize_cache(x), self.decoder.initialize_cache(x)

    def forward(
        self,
        x: torch.Tensor,
        n: int,
        *args
    ) -> tp.Tuple[Tensor, tp.List[Tensor], tp.List[Tensor]]:
        cache = [*args]
        cache_enc: tp.List[Tensor] = cache[:self.encoder.num_cache]
        cache_dec: tp.List[Tensor] = cache[self.encoder.num_cache:]
        x, cache_enc = self.encoder(x, *cache_enc)
        x = self.quantizer(x, n)
        x, cache_dec = self.decoder(x, *cache_dec)
        return x, cache_enc, cache_dec

    def remove_weight_reparameterizations(self):
        if self.norm == "weight_norm":
            for module in self.modules():
                if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                    remove_weight_norm(module)
