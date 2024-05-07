# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
import typing as tp

import numpy as np
import torch
from torch import nn

from modules import ResidualVQ

from . import modules as m


class LMModel(nn.Module):
    """Language Model to estimate probabilities of each codebook entry.
    We predict all codebooks in parallel for a given time step.
    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
    """
    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args:
            indices (torch.Tensor): indices from the previous time step. Indices
                should be 1 + actual index in the codebook. The value 0 is reserved for
                when the index is missing (i.e. first time step). Shape should be
                `[B, n_q, T]`.
            states: state for the streaming decoding.
            offset: offset of the current time step.
        Returns a 3-tuple `(probabilities, new_states, new_offset)` with probabilities
        with a shape `[B, card, n_q, T]`.
        """
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


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
    def __init__(self,
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
                     dim=128, codebook_size=1024, num_quantizers=32
                 )):
        super().__init__()
        channels_vq = vq_kwargs['dim']
        self.encoder = m.SEANetEncoder(channels_audio, channels_vq, channels_enc,
            n_residual_layers, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, true_skip, compress, lstm, causal=True)
        self.decoder = m.SEANetDecoder(channels_audio, channels_vq, channels_dec,
            n_residual_layers, strides, activation, activation_kwargs,
            norm, norm_kwargs, kernel_size, last_kernel_size, residual_kernel_size,
            dilation_base, true_skip, compress, lstm, causal=True,
            final_activation=final_activation)
        if use_vq:
            self.quantizer = ResidualVQ(channel_last=False, **vq_kwargs)
        else:
            self.quantizer = None

        self.sample_rate = sample_rate
        self.channels = channels_audio

    def forward(self, x: torch.Tensor, n: tp.Optional[int] = None) -> torch.Tensor:
        x = self.encoder(x)
        if self.quantizer is not None:
            x, num_replaces, loss_vq = self.quantizer(x, n)
        else:
            x, num_replaces, loss_vq = x, [], torch.zeros(1, dtype=torch.float32, device=x.device)
        x = self.decoder(x)
        #x = torch.tanh(x)
        return x, num_replaces, loss_vq
