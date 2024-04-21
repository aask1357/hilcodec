import typing as tp

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm

from functional import PQMF


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        out = None
        for layer in self.d_convs:
            y = layer(x)
            y = F.leaky_relu(y, 0.2)
            if out is None:
                out = y
            else:
                out = out + y
        y = self.post_conv(out)
        y = F.leaky_relu(y, 0.2)

        return y


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
        self.post_conv = norm_f(Conv1d(
            in_channels=_f[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=3 // 2
        ))

    def forward(self, x):
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)

        return x, fmap


class SBD(torch.nn.Module):
    def __init__(
        self, channels, strides, kernel_sizes, dilations, band_ranges,
        transpose, pqmf_kwargs, f_pqmf_kwargs = {}, segment_size = None,
        use_spectral_norm=False
    ):
        super(SBD, self).__init__()
        self.band_ranges = band_ranges
        self.transpose = transpose
        
        self.pqmf = PQMF(**pqmf_kwargs)
        self.f_pqmf = None
        if True in transpose:
            self.f_pqmf = PQMF(**f_pqmf_kwargs)

        self.discriminators = torch.nn.ModuleList()
        for c, k, d, s, br, tr in zip(
            channels,
            kernel_sizes,
            dilations,
            strides,
            band_ranges,
            transpose
        ):
            if tr:
                segment_dim = segment_size // br[1] - br[0]
            else:
                segment_dim = br[1] - br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=c,
                kernel_size=k,
                dilations=d,
                strides=s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y: Tensor):
        y_ds = []
        fmaps = []
        y_in = self.pqmf(y)

        for d, br, tr in zip(
            self.discriminators,
            self.band_ranges,
            self.transpose
        ):
            if tr:
                y_in_f = self.f_pqmf(y)
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
            else:
                _y_in = y_in[:, br[0]:br[1], :]
            y_d, fmap = d(_y_in)
            y_ds.append(y_d)
            fmaps.extend(fmap)

        return y_ds, fmaps
