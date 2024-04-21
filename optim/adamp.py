"""
Source: https://github.com/clovaai/AdamP (v0.3.0)
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import typing as tp
import math
import torch
from torch import jit, Tensor
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


@jit.script
def projection_channel(p: Tensor, perturb: Tensor, expand_size: tp.List[int], eps: float) -> Tensor:
    p_view = p.view(p.size(0), -1)
    p_n = p / p_view.norm(p=2, dim=1).view(expand_size).add(eps)
    pn_perturb = p_n * perturb
    pnp_view = pn_perturb.view(pn_perturb.size(0), -1)
    perturb -= p_n * pnp_view.sum(dim=1).view(expand_size)
    return perturb


@jit.script
def projection_layer(p: Tensor, perturb: Tensor, expand_size: tp.List[int], eps: float) -> Tensor:
    p_view = p.view(1, -1)
    p_n = p / p_view.norm(p=2, dim=1).view(expand_size).add(eps)
    pn_perturb = p_n * perturb
    pnp_view = pn_perturb.view(1, -1)
    perturb -= p_n * pnp_view.sum(dim=1).view(expand_size)
    return perturb


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False,
                 project_channel=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov,
                        project_channel=project_channel)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for idx, view_func in enumerate([self._channel_view, self._layer_view]):

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                if idx == 0:
                    perturb = projection_channel(p.data, perturb, expand_size, eps)
                else:
                    perturb = projection_layer(p.data, perturb, expand_size, eps)
                return perturb, wd_ratio

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            project_channel = getattr(group, 'project_channel', False)
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if project_channel:
                    expand_size = [-1] + [1] * (len(p.shape) - 1)
                    perturb = projection_channel(p.data, perturb, expand_size, eps)
                    wd_ratio = group['wd_ratio']
                elif len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'],
                                                         group['wd_ratio'], eps)

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
