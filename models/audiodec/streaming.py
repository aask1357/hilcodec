import typing as tp
from typing import Optional, List
import math
import os

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .causal_layers import SConv1d, SConvTranspose1d, CausalConv1d, CausalConvTranspose1d


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


class CausalResidualUnit(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=7,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={}, 
    ):
        super().__init__()
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=bias)
    
    def initialize_cache(self, x: Tensor) -> Tensor:
        return self.conv1.initialize_cache(x)
    
    def forward(self, x, cache):
        y, cache_out = self.conv1(self.activation(x), cache)
        y = self.conv2(self.activation(y))
        return x + y, cache_out


class EncoderBlock(torch.nn.Module):
    """ Encoder block (downsampling) """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilations=(1, 3, 9),
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                CausalResidualUnit(
                    in_channels, 
                    in_channels, 
                    dilation=dilation,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )]
        self.num_res = len(self.res_units)

        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2 * stride),
            stride=stride,
            bias=bias,
        )
        self.num_cache = len(self.initialize_cache(torch.zeros(1, 1, 1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        for block in self.res_units:
            cache_out.append(block.initialize_cache(x))
        return cache_out + [self.conv.initialize_cache(x)]
        
    def forward(self, x, cache_list):
        cache_out = []
        for idx in range(self.num_res):
            x, cache = self.res_units[idx](x, cache_list[idx])
            cache_out.append(cache)
        x, cache = self.conv(x, cache_list[-1])
        cache_out.append(cache)
        return x, cache_out


class Encoder(torch.nn.Module):
    def __init__(self,
        input_channels,
        encode_channels,
        channel_ratios=(2, 4, 8, 16),
        strides=(3, 4, 5, 5),
        kernel_size=7,
        bias=True,
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
        code_dim: int = 64,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = CausalConv1d(
            in_channels=input_channels, 
            out_channels=encode_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            bias=False)

        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = encode_channels * channel_ratios[idx]
            self.conv_blocks += [
                EncoderBlock(
                    in_channels,
                    out_channels, 
                    stride, 
                    bias=bias, 
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels
        self.projector = CausalConv1d(out_channels, code_dim, kernel_size=3, stride=1, bias=False)
        self.num_cache = len(self.initialize_cache(torch.zeros(1, 1, 1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = [self.conv.initialize_cache(x)]
        for block in self.conv_blocks:
            cache_out.extend(block.initialize_cache(x))
        cache_out.append(self.projector.initialize_cache(x))
        return cache_out
    
    def forward(self, x, *args):
        cache_in = [*args]
        cache_out = []
        x, cache = self.conv(x, cache_in[0])
        cache_out.append(cache)
        idx = 1
        for block in self.conv_blocks:
            num_cache = block.num_cache
            x, cache = block(x, cache_in[idx:idx+num_cache])
            cache_out.extend(cache)
            idx += num_cache
        x, cache = self.projector(x, cache_in[-1])
        cache_out.append(cache)
        return x.transpose(1, 2), cache_out


class MultiGroupConv1d(nn.Module):
    """Multi-group convolution module."""

    def __init__(
        self,
        channels=512,
        resblock_kernel_sizes=(3),
        resblock_dilations=[(1, 3, 5)],
        groups=3,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        assert len(resblock_kernel_sizes) == len(resblock_dilations) == 1
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()
        kernel_size = resblock_kernel_sizes[0]
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        for dilation in resblock_dilations[0]:
            self.convs1 += [
                CausalConv1d(
                    in_channels=channels*groups,
                    out_channels=channels*groups,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    CausalConv1d(
                        in_channels=channels*groups,
                        out_channels=channels*groups,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        groups=groups,
                        bias=bias,
                    )
                ]
        self.num_layer = len(self.convs1)
        self.groups = groups
        self.conv_out = nn.Conv1d(
            in_channels=channels*groups, 
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.num_cache = len(self.initialize_cache(torch.zeros(1, 1, 1)))
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = []
        for idx in range(self.num_layer):
            cache_out.append(self.convs1[idx].initialize_cache(x))
            if self.use_additional_convs:
                cache_out.append(self.convs2[idx].initialize_cache(x))
        return cache_out

    def forward(self, x, cache_list):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        cache_out = []
        x = x.repeat(1, self.groups, 1) # (B, n*C, T)
        cache_idx = 0
        for idx in range(self.num_layer):
            xt, cache = self.convs1[idx](self.activation(x), cache_list[cache_idx])
            cache_out.append(cache)
            cache_idx += 1
            if self.use_additional_convs:
                xt, cache = self.convs2[idx](self.activation(xt), cache_list[cache_idx])
                cache_out.append(cache)
                cache_idx += 1
            x = xt + x
        x = self.conv_out(x) # (B, C, T)
        return x, cache_out


class Decoder(nn.Module):
    """HiFiGAN causal generator module."""
    def __init__(
        self,
        in_channels=64,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(5, 5, 4, 3),
        upsample_kernel_sizes=(10, 10, 8, 6),
        resblock_kernel_sizes=[11],
        resblock_dilations=[1, 3, 5],
        groups=3,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        stats="/home/shahn/Documents/codec/AudioDec/stats/symAD_libritts_24000_hop300_clean.npy",
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            groups (int): Number of groups of residual conv
            bias (bool): Whether to add bias parameter in convolution layers.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            stats (str): File name of the statistic file

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # Group conv or MRF
        assert (len(resblock_dilations) == len(resblock_kernel_sizes) == 1) and (groups > 1), \
            "MRF Not implemented. Use Grouped Convolution instead"
        
        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.input_conv = CausalConv1d(
            in_channels,
            channels,
            kernel_size,
            stride=1,
        )
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.activation_upsamples = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                CausalConvTranspose1d(
                    channels // (2 ** i),
                    channels // (2 ** (i + 1)),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_scales[i],
                )
            ]
            self.blocks += [
                MultiGroupConv1d(
                    channels=channels // (2 ** (i + 1)),
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilations,
                    groups=groups,
                    bias=bias,
                    use_additional_convs=use_additional_convs,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
        self.activation_output1 = nn.LeakyReLU()
        self.activation_output2 = nn.Tanh()
        self.output_conv = CausalConv1d(
            channels // (2 ** (i + 1)),
            out_channels,
            kernel_size,
            stride=1,
        )

        # load stats
        if stats is not None:
            self.register_stats(stats)
            self.norm = True
        else:
            self.norm = False

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_out: tp.List[Tensor] = [self.input_conv.initialize_cache(x)]
        for i in range(self.num_upsamples):
            cache_out.append(self.upsamples[i].initialize_cache(x))
            cache_out.extend(self.blocks[i].initialize_cache(x))
        cache_out.append(self.output_conv.initialize_cache(x))
        return cache_out
    
    def forward(self, c, *args):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        if self.norm:
            c = (c - self.mean) / self.scale
        c = c.transpose(1, 2)

        cache_in: tp.List[Tensor] = [*args]
        c, cache = self.input_conv(c, cache_in[0])
        cache_out = [cache]
        idx = 1
        for i in range(self.num_upsamples):
            c, cache = self.upsamples[i](self.activation_upsamples(c), cache_in[idx])
            cache_out.append(cache)
            num_cache = self.blocks[i].num_cache
            c, cache = self.blocks[i](c, cache_in[idx+1:idx+1+num_cache])
            cache_out.extend(cache)
            idx += num_cache + 1
        c, cache = self.output_conv(self.activation_output1(c), cache_in[-1])
        cache_out.append(cache)
        c = self.activation_output2(c)

        return c, cache_out

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(
                m, nn.ConvTranspose1d
            ):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        assert os.path.exists(stats), f"Stats {stats} does not exist!"
        mean = np.load(stats)[0].reshape(-1)
        scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())


class AudioDec(torch.nn.Module):
    """AudioDec generator."""
    def __init__(
        self,
        # encoder
        input_channels=1,
        encode_channels=32,
        enc_ratios=(2, 4, 8, 16),
        enc_strides=(3, 4, 5, 5),
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
        # quantizer
        code_dim=64,
        codebook_num=8,
        codebook_size=1024,
        # decoder
        output_channels=1,
        decode_channels=512,
        dec_strides=(5, 5, 4, 3),
        dec_kernel_sizes=(10, 10, 8, 6),
        kernel_size=7,
        resblock_kernel_sizes=[11],
        resblock_dilations=[[1,3,5]],
    ):
        super().__init__()
        self.input_channels = input_channels

        self.encoder = Encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=7,
            bias=True,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
        )
        self.quantizer = ResidualVQ(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size,
        )
        self.dequantizer = Dequantizer(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size,
        )
        self.decoder = Decoder(
            in_channels=code_dim,
            out_channels=output_channels,
            channels=decode_channels,
            kernel_size=kernel_size,
            upsample_scales=dec_strides,
            upsample_kernel_sizes=dec_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            groups=3,
            bias=True,
            use_additional_convs=True,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.1},
            use_weight_norm=True,
            stats="/home/shahn/Documents/codec/AudioDec/stats/symAD_libritts_24000_hop300_clean.npy",
        )

    def initialize_cache(self, x):
        cache_enc = self.encoder.initialize_cache(x)
        cache_dec = self.decoder.initialize_cache(x)
        return cache_enc, cache_dec

if __name__=="__main__":
    model = Encodec()
    x = torch.randn(1, 1, 1000)
    cache_enc, cache_dec = model.initialize_cache(x)
    model(x, 8, *cache_enc, *cache_dec)