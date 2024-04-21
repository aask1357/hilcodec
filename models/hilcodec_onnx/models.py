# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import typing as tp

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils import remove_weight_norm

from modules.weight_norm_all import remove_weight_norm_all
from utils import verbose
from . import modules as m
from .vector_quantize import ResidualVQ


Array = tp.Union[np.ndarray, list]


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
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
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        skip: str = "1x1",
        compress: int = 2,
        lstm: int = 2,
        final_activation: tp.Optional[str] = None,
        use_vq: bool = True,    # deprecated
        vq: str = "ResidualVQ", # "" / "ResidualVQ" / "ResidualGainShapeVQ" / "ResidualGainResidualShapeVQ"
        vq_kwargs: tp.Dict[str, tp.Any] = {},
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        encoder_l2norm: bool = False,
        bias: bool = True,
        spec: str = "stft",          # dct or stft
        spec_layer: str = "1x1",    # 1x1 or scale or channelwise_scale
        spec_compression: str = "", # "" or "log" or float(0~1)
        spec_learnable: bool = False,
        mu: bool = False,
        pad_mode: str = "reflect",
        causal: bool = True,
    ):
        assert spec in ["stft", ""]
        assert spec_layer in ["1x1", "scale", "channelwise_scale", "1x1_zero", "1x1_div2"]
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
        self.encoder = m.SEANetEncoder(channels_audio, channels_vq, channels_enc,
            n_residual_enc, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, skip, compress, lstm, causal=causal,
            act_all=act_all, expansion=expansion, groups=groups, l2norm=encoder_l2norm,
            bias=bias, spec=spec, spec_layer=spec_layer, spec_compression=spec_compression,
            res_scale=res_scale_enc, mu=mu, pad_mode=pad_mode,
            spec_learnable=spec_learnable,)
        self.decoder = m.SEANetDecoder(channels_audio, channels_vq, channels_dec,
            n_residual_dec, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, skip, compress, lstm, causal=causal,
            final_activation=final_activation,
            act_all=act_all, expansion=expansion, groups=groups, bias=bias,
            res_scale=res_scale_dec, pad_mode=pad_mode,)
        if use_vq:
            if vq == "ResidualVQ":
                self.quantizer = ResidualVQ(channel_last=False, **vq_kwargs)
            else:
                raise ValueError(f"Unknown vq: {vq}")
        else:
            self.quantizer = None

        self.sample_rate = sample_rate
        self.channels = channels_audio

    def forward(self, x: Tensor, n: tp.Optional[int] = None) -> tp.Tuple[Tensor, Array, Tensor]:
        x = self.encoder(x)
        if self.quantizer is not None:
            x, num_replaces, loss_vq = self.quantizer(x, n)
        else:
            num_replaces, loss_vq = [], torch.zeros(1, dtype=torch.float32, device=x.device)
        x = self.decoder(x)
        return x.float(), num_replaces, loss_vq

    def remove_weight_reparameterizations(self):
        if self.norm == "weight_norm":
            for module in self.modules():
                if isinstance(module, m.NormConv1d):
                    module.conv = remove_weight_norm(module.conv)
        elif self.norm == "weight_norm_all":
            for module in self.modules():
                if isinstance(module, m.NormConv1d):
                    module.conv = remove_weight_norm_all(module.conv)
        for module in self.modules():
            if hasattr(module, merge_scaling):
                module.merge_scaling()
