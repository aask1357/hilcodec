import typing as tp
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from einops import rearrange, repeat


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))


def sample_vectors(samples: Tensor, num: int) -> Tensor:
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
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.ema_num_threshold = ema_num_threshold
        self.ema_num_initial = ema_num_initial

        self.initted = not kmeans_init
        self.register_buffer('embed', embed)
        self.register_buffer('ema_embed', embed.clone() * ema_num_initial)
        self.register_buffer('ema_num', torch.ones(codebook_size) * ema_num_initial)
        self.embed: Tensor
        self.ema_num: Tensor
        self.ema_embed: Tensor
        
        self.distributed = dist.is_initialized() and (dist.get_world_size() > 1)
    
    def get_extra_state(self) -> tp.Dict[str, bool]:
        return {"initted": self.initted}
    
    def set_extra_state(self, state: tp.Dict[str, tp.Any]) -> None:
        self.initted = state["initted"]
        
    def init_embed_(self, data: Tensor) -> None:
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        if self.distributed:
            dist.broadcast(embed, 0)
        self.embed.data.copy_(embed)
        self.ema_embed.data.copy_(embed * self.ema_num_initial)
        self.ema_num.data.fill_(self.ema_num_initial)
        self.initted = True

    def replace(self, samples: Tensor, mask: Tensor) -> int:
        idx = torch.nonzero(mask).squeeze(1)
        new_embed = sample_vectors(samples, idx.size(0)).float()
        if self.distributed:
            dist.broadcast(new_embed, 0)
        self.embed.data[idx, :] = new_embed

        self.ema_embed.data[idx, :] = new_embed * self.ema_num_initial
        self.ema_num[idx].fill_(self.ema_num_initial)
        return idx.size(0)

    def expire_codes_(self, batch_samples: Tensor) -> int:
        # batch_samples: [Batch x Time, Channel]
        if self.ema_num_threshold == 0.0:
            return 0

        expired_codes = self.ema_num < self.ema_num_threshold   # [codebook_size]
        if not torch.any(expired_codes):
            return 0
        return self.replace(batch_samples, mask=expired_codes)

    @torch.no_grad()
    def forward(self, x: Tensor) -> tp.Tuple[Tensor, int]:
        # x.dtype: float16 / embed.dtype: float32 / device: cuda{rank}
        shape, dtype = x.shape, self.embed.dtype
        flatten = rearrange(x, '... d -> (...) d')      # [Batch x Time, Channel]
        embed = self.embed.t()                          # [Channel, codebook_size]

        if not self.initted:
            self.init_embed_(flatten)

        # distance: [Batch x Time, codebook_size]
        # distance[i,j] = ||flatten[i, :] - embed[:, j||^2_2 = sum_c{(flatten[i,c]-embed[c,j])^2}
        distance = (
            # flatten.pow(2).sum(1, keepdim=True)   # constant w.r.t. embed
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = distance.min(dim = -1).indices      # [Batch x Time]
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)     # [Batch x Time, codebook_size]
        embed_ind = embed_ind.view(*shape[:-1])         # [Batch, Time]
        quantize = F.embedding(embed_ind, self.embed)   # [Batch, Time, Channel]

        if self.training:
            with autocast(enabled=False):
                num_curr = embed_onehot.sum(dim=0)              # [codebook_size]
                embed_curr = embed_onehot.t() @ flatten.float() # [codebook_size, Channel]
                if self.distributed:
                    # Concatenate multiple tensors before all_reduce for speedup
                    codebook_size, channel = embed_curr.shape
                    bucket = torch.cat([num_curr, embed_curr.view(-1)])
                    dist.all_reduce(bucket)
                    num_curr = bucket[:codebook_size]
                    embed_curr = bucket[codebook_size:].reshape(codebook_size, channel)

                ema_inplace(self.ema_num, num_curr, self.decay)
                ema_inplace(self.ema_embed, embed_curr, self.decay)

                embed_normalized = self.ema_embed / self.ema_num.unsqueeze(1)
                self.embed.data.copy_(embed_normalized)
                num_replace = self.expire_codes_(flatten)
        else:
            num_replace = 0

        return quantize, num_replace


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers: int,
        dropout: bool = False,
        dropout_index: tp.Optional[tp.List[int]] = None,
        channel_last: bool = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EuclideanCodebook(**kwargs) for _ in range(num_quantizers)
        ])
        self.dropout = dropout
        if dropout_index is None:
            dropout_index = list(range(1, num_quantizers + 1))
        self.dropout_index = dropout_index      # index of VQ layers to choose when dropout==True
        self.channel_last = channel_last

    def forward(
        self,
        x: Tensor,
        n: tp.Optional[int] = None
    ) -> tp.Tuple[Tensor, np.ndarray, Tensor]:
        if not self.channel_last:
            residual = x.transpose(1, 2).detach()   # [B, C, T] -> [B, T, C]
        else:
            residual = x.detach()
        num_replaces = np.zeros(len(self.layers), dtype=np.int64)

        if n is not None:
            assert 1 <= n <= len(self.layers), \
                f"'n' must be in range of 1 <= n <= {len(self.layers)}"
            high = n
        elif self.training and self.dropout:
            high = random.sample(self.dropout_index, 1)[0]
        else:
            high = len(self.layers)

        for idx, layer in enumerate(self.layers[:high]):
            quantized, num_replace = layer(residual)
            num_replaces[idx] = num_replace
            residual = residual - quantized
            if idx == 0:
                quantized_out = quantized
            else:
                quantized_out = quantized_out + quantized
        quantized_out: Tensor
        
        if not self.channel_last:
            quantized_out = quantized_out.transpose(1, 2)    # [B, T, C] -> [B, C, T]
        
        loss = F.mse_loss(x, quantized_out)
        if self.training:
            quantized_out = quantized_out + x - x.detach()
            
        
        return quantized_out, num_replaces, loss
