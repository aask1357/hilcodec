import random
from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
import numpy as np
from torch.cuda.amp import autocast


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))


def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            distance = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            distance = -(diffs ** 2).sum(dim = -1)

        buckets = distance.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# distance types

class EuclideanCodebook(nn.Module):
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
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.ema_num_threshold = ema_num_threshold
        self.ema_num_initial = ema_num_initial

        self.register_buffer('initted', Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)
        self.register_buffer('ema_embed', embed.clone() * ema_num_initial)
        self.register_buffer('ema_num', torch.ones(codebook_size) * ema_num_initial)
        self.initted: Tensor
        self.embed: Tensor
        self.ema_embed: Tensor
        self.ema_num: Tensor

        self.distributed = dist.is_initialized() and (dist.get_world_size() > 1)
        
    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        if self.distributed:
            dist.broadcast(embed, 0)
        self.embed.data.copy_(embed)
        self.ema_embed.data.copy_(embed * self.ema_num_initial)
        self.ema_num.data.fill_(self.ema_num_initial)
        self.initted.data.copy_(Tensor([True]))

    def replace(self, samples, mask) -> int:
        idx = torch.nonzero(mask).squeeze(1)
        new_embed = sample_vectors(samples, idx.size(0)).detach().float()
        if self.distributed:
            dist.broadcast(new_embed, 0)
        self.embed.data[idx, :] = new_embed

        self.ema_embed.data[idx, :] = new_embed * self.ema_num_initial
        self.ema_num.data[idx] = self.ema_num_initial
        return idx.size(0)

    def expire_codes_(self, batch_samples):
        if self.ema_num_threshold == 0.0:
            return 0

        expired_codes = self.ema_num < self.ema_num_threshold
        if not torch.any(expired_codes):
            return 0
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        return self.replace(batch_samples, mask = expired_codes)

    @torch.no_grad()
    def forward(self, x):
        # x.dtype: float16 / embed.dtype: float32 / device: cuda{rank}
        shape, dtype, device = x.shape, self.embed.dtype, x.device
        flatten = rearrange(x, '... d -> (...) d')      # [Batch x Time, Channel]
        embed = self.embed.t()                          # [Channel, codebook_size]

        if not self.initted:
            self.init_embed_(flatten)

        # distance: [Batch x Time, codebook_size]
        distance = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = distance.max(dim = -1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            with autocast(enabled=False):
                if self.distributed:
                    # Concatenate multiple tensors before all_reduce for speedup
                    ema_num_numel, ema_embed_numel = self.ema_num.numel(), self.ema_embed.numel()
                    bucket = torch.empty(ema_num_numel + ema_embed_numel,
                        dtype=dtype, device=device)
                    bucket[:ema_num_numel] = embed_onehot.sum(dim=0)
                    bucket[ema_num_numel:] = (flatten.t().float() @ embed_onehot).view(-1)
                    dist.all_reduce(bucket)
                    ema_num_new = bucket[:ema_num_numel]
                    ema_embed_new = bucket[ema_num_numel:].reshape(self.ema_embed.size(1), self.ema_embed.size(0))
                else:
                    ema_num_new = embed_onehot.sum(dim=0)
                    ema_embed_new = flatten.t().float() @ embed_onehot

                ema_inplace(self.ema_num, ema_num_new, self.decay)
                ema_inplace(self.ema_embed, ema_embed_new.t(), self.decay)

                # Make sure that ema_num > eps (Not needed if use random reinitialization)
                if self.ema_num_threshold <= 0.0:
                    ema_num = laplace_smoothing(
                        self.ema_num, self.codebook_size, self.eps
                    ) * self.ema_num.sum()
                else:
                    ema_num = self.ema_num

                embed_normalized = self.ema_embed / ema_num.unsqueeze(1)
                self.embed.data.copy_(embed_normalized)
                num_replace = self.expire_codes_(x)
        else:
            num_replace = 0

        return quantize, num_replace


class ShapeGainCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        shape_codebook_size: int,
        gain_codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 20,
        decay: float = 0.99,
        eps: float = 1e-5,
        ema_num_threshold: float = 0.0,
        ema_num_initial: float = 1.0,
        log_gain: bool = True,
    ):
        super().__init__()
        self.decay = decay

        self.shape_codebook_size = shape_codebook_size
        self.gain_codebook_size = gain_codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.ema_num_threshold = ema_num_threshold
        self.ema_num_initial = ema_num_initial
        self.gain_num_ratio = shape_codebook_size / gain_codebook_size
        self.log_gain = log_gain

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('shape', torch.randn(self.shape_codebook_size, dim))
        self.register_buffer('shape_num', torch.ones(self.shape_codebook_size) * ema_num_initial)
        self.register_buffer('gain', torch.rand(gain_codebook_size) * 0.9 + 1.0)        # Uniform(0.1, 1.9)
        self.register_buffer('gain_num', torch.ones(gain_codebook_size) * ema_num_initial * self.gain_num_ratio)

        self.distributed = dist.is_initialized() and (dist.get_world_size() > 1)
        
    def init_embed_(self, data):
        # data: [Batch x Time, Channel]
        # shape
        shape_data = l2norm(data)
        shape, cluster_size = kmeans(shape_data, self.shape_codebook_size, self.kmeans_iters)

        # gain = dot(data, shape_quantize)
        distance = data @ shape.t()
        shape_ind = distance.max(dim = -1).indices
        shape_quantize = F.embedding(shape_ind, shape)
        gain_data = (data * shape_quantize).sum(dim = 1, keepdim = True)
        if self.log_gain:
            gain_data = gain_data.clamp(min=self.eps).log()
        gain, cluster_size = kmeans(gain_data, self.gain_codebook_size, self.kmeans_iters)
        gain = gain.squeeze()
        
        if self.distributed:
            gain_numel = gain.numel()
            bucket = torch.cat(gain.view(-1), shape.view(-1))
            dist.broadcast(bucket, 0)
            gain, shape = bucket[:gain_numel], bucket[gain_numel:].reshape(self.shape_codebook_size, -1)
        
        self.gain.data.copy_(gain)
        self.gain_num.data.fill_(self.ema_num_initial * self.gain_num_ratio)
        self.shape.data.copy_(shape)
        self.shape_num.data.fill_(self.ema_num_initial)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace_shape(self, samples, mask) -> int:
        idx = torch.nonzero(mask).squeeze(1)
        shape_new = sample_vectors(samples, idx.size(0)).detach().float()
        if self.distributed:
            dist.broadcast(shape_new, 0)
        self.shape.data[idx, :] = shape_new
        self.shape_num[idx].fill_(self.ema_num_initial)
        return idx.size(0)

    def replace_gain(self, samples, mask) -> int:
        idx = torch.nonzero(mask).squeeze(1)
        gain_new = sample_vectors(samples, idx.size(0)).detach().float()
        if self.distributed:
            dist.broadcast(gain_new, 0)
        self.gain.data[idx] = gain_new
        self.gain_num[idx].fill_(self.ema_num_initial * self.gain_num_ratio)
        return idx.size(0)

    def expire_codes_(self, flatten: torch.tensor, gain_flatten: torch.tensor) -> Tuple[int, int]:
        if self.ema_num_threshold == 0.0:
            return 0

        # shape
        expired_codes = self.shape_num < self.ema_num_threshold
        if torch.any(expired_codes):
            shape_flatten = l2norm(flatten)
            shape_num_replace = self.replace_shape(shape_flatten, mask = expired_codes)
        else:
            shape_num_replace = 0
        
        # gain
        expired_codes = self.gain_num < self.ema_num_threshold * self.gain_num_ratio
        if torch.any(expired_codes):
            gain_num_replace = self.replace_gain(gain_flatten, mask = expired_codes)
        else:
            gain_num_replace = 0
        return shape_num_replace, gain_num_replace

    @torch.no_grad()
    def forward(self, x: torch.tensor):
        # x: [Batch(B), Time(T), Channel(C)]
        # self.shape: [shape_codebook_size(scs), Channel(C)]
        # self.gain: [gain_codebook_size(gcs)]
        x_shape, dtype = x.shape, self.shape.dtype          # x.dtype: float16 / embed.dtype: float32
        flatten = rearrange(x, '... d -> (...) d').float()  # [Batch x Time(BT), Channel(C)]
        
        with autocast(enabled=False):
            if not self.initted:
                self.init_embed_(flatten)

            # shape
            distance = flatten @ self.shape.t()                                         # [BT, C] x [C, scs] = [BT, scs]
            shape_ind = distance.max(dim = -1).indices                                  # [BT]
            shape_onehot = F.one_hot(shape_ind, self.shape_codebook_size).type(dtype)   # [BT, scs]
            #shape_ind = shape_ind.view(*x_shape[:-1])
            shape_quantize = F.embedding(shape_ind, self.shape)                         # [BT, C]

            # gain
            gain_flatten = (flatten * shape_quantize).sum(dim = 1, keepdim = True)      # [BT, 1]
            if self.log_gain:
                gain_flatten = gain_flatten.clamp(min=self.eps).log()
            gain = self.gain.unsqueeze(0)       # [1, gcs]
            distance = -(                       # [BT, 1] x [1, gcs] = [BT, gcs]
                gain_flatten.pow(2)
                - 2 * gain_flatten @ gain
                + gain.pow(2)
            )
            gain_ind = distance.max(dim = -1).indices                                   # [BT]
            gain_onehot = F.one_hot(gain_ind, self.gain_codebook_size).type(dtype)      # [BT, gcs]
            gain_ind = gain_ind.view(*x_shape[:-1])                                     # [B, T, gcs]
            gain_quantize = F.embedding(gain_ind, self.gain)                            # [B, T]
            shape_quantize = shape_quantize.view(*x_shape[:-1], -1)                     # [B, T, C]

            if self.training:
                shape_num_new = shape_onehot.sum(dim = 0)
                shape_new = flatten.t() @ shape_onehot
                gain_num_new = gain_onehot.sum(dim = 0)
                gain_new = (gain_flatten.t() @ gain_onehot).squeeze()
                if self.distributed:
                    # Concatenate multiple tensors before all_reduce for speedup
                    bucket = torch.cat(gain_num_new, shape_num_new, gain_new, shape_new.view(-1))
                    dist.all_reduce(bucket)

                    end = self.gain_codebook_size
                    gain_num_new = bucket[:end]

                    start, end = end, end + self.shape_codebook_size
                    shape_num_new = bucket[self.gain_codebook_size:end]

                    start, end = end, end + self.gain_codebook_size
                    gain_new = bucket[start:end]

                    start = end
                    shape_new = bucket[start:].reshape(self.shape.size(1), self.shape.size(0))
                shape_new.div_(shape_new.norm(p=2, dim=0, keepdim=True).clamp(min=self.eps))
                gain_new.div_(gain_num_new.clamp(min=self.eps))

                ema_inplace(self.gain_num, gain_num_new, self.decay)
                ema_inplace(self.gain, gain_new, self.decay)
                ema_inplace(self.shape_num, shape_num_new, self.decay)
                ema_inplace(self.shape, shape_new.t(), self.decay)
                shape_normalized = l2norm(self.shape)
                self.shape.data.copy_(shape_normalized)
                
                num_replace = self.expire_codes_(flatten, gain_flatten.squeeze())
            else:
                num_replace = (0, 0)
            
            if self.log_gain:
                gain_quantize = gain_quantize.exp()
            quantize = gain_quantize.unsqueeze(-1) * shape_quantize

        return quantize, num_replace


# main class
class VectorQuantize(nn.Module):
    def __init__(
        self,
        commitment: float = 1.,
        use_shape_gain: bool = False,
        channel_last: bool = False,
        gradient_flow: bool = True,
        **kwargs
    ):
        super().__init__()
        self.commitment = commitment
        self.use_shape_gain = use_shape_gain
        self.channel_last = channel_last

        if use_shape_gain:
            codebook_class = ShapeGainCodebook
        else:
            codebook_class = EuclideanCodebook

        self._codebook = codebook_class(
            **kwargs
        )
        self.gradient_flow = gradient_flow

    def forward(self, x, calculate_commitment_loss: bool = False):
        if not self.channel_last:
            # [batch, channel, time] -> [batch, time, channel]
            x = rearrange(x, 'b c t -> b t c')
        
        quantize, num_replace = self._codebook(x)

        if calculate_commitment_loss:
            commit_loss = F.mse_loss(quantize.detach(), x) * self.commitment
        else:
            commit_loss = None

        if self.gradient_flow and self.training:
            quantize = x + quantize - x.detach()
        
        if not self.channel_last:
            # [batch, time, channel] -> [batch, channel, time]
            quantize = rearrange(quantize, 'b t c -> b c t')

        return quantize, num_replace, commit_loss


class ResidualShapeGainVQ(nn.Module):
    """ Residual VQ: Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    ShapeGain VQ: First residual VQ quantize the gain, and the rest VQs quantize the shape """
    def __init__(
        self,
        num_quantizers: int,
        dropout: bool = False,
        dropout_index: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantize(gradient_flow=False, **kwargs) for idx in range(num_quantizers)
        ])
        self.dropout = dropout
        if dropout_index is None:
            dropout_index = list(range(1, num_quantizers + 1))
        self.dropout_index = dropout_index      # index of VQ layers to choose when dropout==True
        self.use_shape_gain = self.layers[0].use_shape_gain

    def forward(self, x, n=None):
        quantized_out = 0.
        k = 2 if self.use_shape_gain else 1
        num_replaces = np.zeros(k * len(self.layers), dtype=np.int64)

        if n is not None:
            assert 1 <= n <= len(self.layers), \
                f"'n' must be in range of 1 <= n <= {len(self.layers)}"
            high = n
        elif self.training and self.dropout:
            #high = random.randint(0, len(self.layers)-1)    # 0 <= high <= len(self.layers)-1
            high = random.sample(self.dropout_index, 1)[0]
        else:
            high = len(self.layers)

        residual = x.detach()
        for idx, layer in enumerate(self.layers[:high]):
            quantized, num_replace, _ = layer(residual, calculate_commitment_loss=False)
            num_replaces[k * idx : k * (idx + 1)] = num_replace
            residual = residual - quantized
            quantized_out = quantized_out + quantized
        
        loss = F.mse_loss(x, quantized_out)
        if self.training:
            quantized_out = quantized_out + x - x.detach()
        
        return quantized_out, num_replaces, loss


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
            VectorQuantize(gradient_flow=False, **kwargs) for idx in range(num_quantizers)
        ])
        self.dropout = dropout
        if dropout_index is None:
            dropout_index = list(range(1, num_quantizers + 1))
        self.dropout_index = dropout_index      # index of VQ layers to choose when dropout==True
        self.use_shape_gain = self.layers[0].use_shape_gain

    def forward(self, x, n=None):
        quantized_out = 0.
        k = 2 if self.use_shape_gain else 1
        num_replaces = np.zeros(k * len(self.layers), dtype=np.int64)

        if n is not None:
            assert 1 <= n <= len(self.layers), \
                f"'n' must be in range of 1 <= n <= {len(self.layers)}"
            high = n
        elif self.training and self.dropout:
            #high = random.randint(0, len(self.layers)-1)    # 0 <= high <= len(self.layers)-1
            high = random.sample(self.dropout_index, 1)[0]
        else:
            high = len(self.layers)

        residual = x.detach()
        for idx, layer in enumerate(self.layers[:high]):
            quantized, num_replace, _ = layer(residual, calculate_commitment_loss=False)
            num_replaces[k * idx : k * (idx + 1)] = num_replace
            residual = residual - quantized
            quantized_out = quantized_out + quantized
        
        loss = F.mse_loss(x, quantized_out)
        if self.training:
            quantized_out = quantized_out + x - x.detach()
        
        return quantized_out, num_replaces, loss
