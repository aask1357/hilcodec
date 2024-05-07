import typing as tp
from typing import Optional, List
import math

import numpy as np
import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .causal_layers import SConv1d, SConvTranspose1d, CausalConv1d, CausalConvTranspose1d, SLSTM


class EuclideanCodebook(nn.Module):
    __constants__ = ['codebook_size', 'eps', 'ema_num_initial']
    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 1024,
        kmeans_init: bool = False,
        kmeans_iters: int = 20,
        decay: float = 0.8,
        eps: float = 1e-7,
        ema_num_threshold: float = 0.0,
        ema_num_initial: float = 1.0,
    ):
        super().__init__()
        self.decay = decay
        embed = torch.randn(codebook_size, dim)

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
        flatten = x.reshape(B * T, C, 1)    # [BxT, Channel, 1]
        embed = self.embed.t().unsqueeze(0) # [1, Channel, codebook_size]

        # distance: [Batch x Time, codebook_size]
        # distance = -(
        #     flatten.pow(2).sum(1, keepdim=True)
        #     - 2 * flatten @ embed
        #     + embed.pow(2).sum(0, keepdim=True)
        # )
        distance = -(flatten - embed).pow(2).sum(1) # [BxT, codebook_size]
        
        embed_ind = distance.max(dim = -1).indices
        embed_ind = embed_ind.view(B, T)
        quantized = F.embedding(embed_ind, self.embed)  # [B, T, C]

        return quantized, embed_ind
    
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantized = F.embedding(embed_ind, self.embed)
        return quantized


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers: int = 16,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EuclideanCodebook(**kwargs) for _ in range(num_quantizers)
        ])

    def forward(self, x: Tensor, n: int) -> Tensor:
        # x: [B, T, C]
        residual = x
        quantized_out = torch.zeros_like(residual)
        indices = []
        for layer in self.layers[:n]:
            quantized, index = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            indices.append(index)
        
        return torch.stack(indices, dim=0)  # [n, B, T]


class EuclideanCodebookDeq(nn.Module):
    __constants__ = ['codebook_size', 'eps', 'ema_num_initial']
    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 1024,
        kmeans_init: bool = False,
        kmeans_iters: int = 20,
        decay: float = 0.8,
        eps: float = 1e-7,
        ema_num_threshold: float = 0.0,
        ema_num_initial: float = 1.0,
    ):
        super().__init__()
        self.decay = decay
        embed = torch.randn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.eps = eps
        self.ema_num_initial = ema_num_initial

        self.register_buffer('embed', embed)
        self.register_buffer('ema_num', torch.ones(codebook_size) * ema_num_initial)
        self.embed: Tensor
        self.ema_num: Tensor

    def forward(self, embed_ind: Tensor) -> Tensor:
        quantized = F.embedding(embed_ind, self.embed)
        return quantized


class Dequantizer(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers: int = 16,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EuclideanCodebookDeq(**kwargs) for _ in range(num_quantizers)
        ])

    def forward(self, indices: Tensor, n: int) -> Tensor:
        # indices: [n, B, T]
        out = torch.zeros(1, dtype=torch.float32, device=indices.device)
        for i in range(n):
            layer = self.layers[i]
            index = indices[i]
            quantized = layer(index)
            out = out + quantized
        
        return out      # [B, C, T]


class ResBlock(nn.Module):
    def __init__(
        self, dim: int, kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [3, 1], activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        compress: int = 2, true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm)
        self.num_cache = len(self.initialize_cache(torch.zeros(1, 1, 1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache: tp.List[Tensor] = []
        for block in self.block:
            if isinstance(block, CausalConv1d):
                cache.append(block.initialize_cache(x))
        return cache

    def forward(
        self,
        x: Tensor,
        cache: tp.List[Tensor]
    ) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        new_cache: tp.List[Tensor] = []
        skip = x
        idx = 0
        for block in self.block:
            if isinstance(block, CausalConv1d):
                x, cache_idx = block(x, cache[idx])
                new_cache.append(cache_idx)
                idx += 1
            else:
                x = block(x)
        return self.shortcut(skip) + x, new_cache


class Encoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, true_skip: bool = False, compress: int = 2, lstm: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        
        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            SConv1d(
                channels, mult * n_filters, kernel_size, norm=norm,
                norm_kwargs=norm_params,
            )
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [ResBlock(
                    mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                    dilations=[dilation_base ** j, 1],
                    norm=norm, norm_params=norm_params,
                    activation=activation, activation_params=activation_params,
                    compress=compress,
                    true_skip=true_skip
                )]

            # Add downsampling layers
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,),
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm,
                    norm_kwargs=norm_params)
        ]

        self.model = nn.Sequential(*model)
        self.num_cache = len(self.initialize_cache(torch.zeros(1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        for block in self.model:
            if isinstance(block, CausalConv1d):
                cache_out.append(block.initialize_cache(x))
            elif isinstance(block ,SLSTM):
                cache = block.initialize_cache(x)
                cache_out.append(cache[0])
                cache_out.append(cache[1])
            elif isinstance(block, ResBlock):
                cache_out.extend(block.initialize_cache(x))
        return cache_out

    def forward(self, x: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # x: [B, 1, T]
        cache_in: tp.List[Tensor] = [*args]
        cache_out: tp.List[Tensor] = []
        
        idx = 0
        for block in self.model:
            if isinstance(block, CausalConv1d):
                x, cache = block(x, cache_in[idx])
                cache_out.append(cache)
                idx += 1
            elif isinstance(block, SLSTM):
                x, cache = block(x, cache_in[idx:idx+2])
                cache_out.append(cache[0])
                cache_out.append(cache[1])
                idx += 2
            elif isinstance(block, ResBlock):
                num_cache = block.num_cache
                x, cache = block(x, cache_in[idx:idx+num_cache])
                cache_out.extend(cache)
                idx += num_cache
            else:
                x = block(x)
        return x.transpose(1, 2), cache_out


class Decoder(nn.Module):
    def __init__(
        self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
        n_residual_layers: int = 1, ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3,
        dilation_base: int = 2, true_skip: bool = False, compress: int = 2, lstm: int = 2,
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
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
            SConv1d(
                dimension, mult * n_filters, kernel_size, norm=norm,
                norm_kwargs=norm_params,)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                act(**activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [ResBlock(
                    mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                    dilations=[dilation_base ** j, 1],
                    activation=activation, activation_params=activation_params,
                    norm=norm, norm_params=norm_params, compress=compress, true_skip=true_skip
                )]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            SConv1d(
                n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        for block in self.model:
            if isinstance(block, (CausalConv1d, CausalConvTranspose1d)):
                cache_out.append(block.initialize_cache(x))
            elif isinstance(block, SLSTM):
                cache = block.initialize_cache(x)
                cache_out.append(cache[0])
                cache_out.append(cache[1])
            elif isinstance(block, ResBlock):
                cache_out.extend(block.initialize_cache(x))
        return cache_out

    def forward(self, x: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # x: [B, T', C] / out: [B, 1, T]
        x = x.transpose(1, 2)
        cache_in: tp.List[Tensor] = [*args]
        cache_out: tp.List[Tensor] = []
        
        idx = 0
        for block in self.model:
            if isinstance(block, (CausalConv1d, CausalConvTranspose1d)):
                x, cache = block(x, cache_in[idx])
                cache_out.append(cache)
                idx += 1
            elif isinstance(block, SLSTM):
                x, cache = block(x, cache_in[idx:idx+2])
                cache_out.append(cache[0])
                cache_out.append(cache[1])
                idx += 2
            elif isinstance(block, ResBlock):
                num_cache = block.num_cache
                x, cache = block(x, cache_in[idx:idx+num_cache])
                cache_out.extend(cache)
                idx += num_cache
            else:
                x = block(x)
        return x, cache_out


class Encodec(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24_000,
        channels_audio: int = 1,
        channels_enc: int = 32,
        channels_dec: int = 32,
        n_residual_layers: int = 1,
        strides: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_kwargs: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        final_activation: tp.Optional[str] = None,
        use_vq: bool = True,
        vq_kwargs: tp.Dict[str, tp.Any] = dict(
            dim=128, codebook_size=1024, num_quantizers=32)
    ):
        assert use_vq
        super().__init__()
        self.sample_rate = sample_rate
        self.norm = norm
        channels_vq = vq_kwargs['dim']
        self.encoder = Encoder(channels_audio, channels_vq, channels_enc,
            n_residual_layers, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, true_skip, compress, lstm,)
        self.decoder = Decoder(channels_audio, channels_vq, channels_dec,
            n_residual_layers, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, true_skip, compress, lstm,
            final_activation=final_activation)
        self.quantizer = ResidualVQ(**vq_kwargs)
        self.dequantizer = Dequantizer(**vq_kwargs)
    
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
        x = self.dequantizer(x, n)
        x, cache_dec = self.decoder(x, *cache_dec)
        return x, cache_enc, cache_dec

    def remove_weight_reparameterizations(self):
        if self.norm == "weight_norm":
            for module in self.modules():
                if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                    remove_weight_norm(module)


if __name__=="__main__":
    model = Encodec()
    x = torch.randn(1, 1, 1000)
    cache_enc, cache_dec = model.initialize_cache(x)
    model(x, 8, *cache_enc, *cache_dec)