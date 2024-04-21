import typing as tp

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch import autograd
from torch import distributed as dist


class Balancer:
    def __init__(
        self, weights: tp.Dict[str, float], others: tp.List[str],
        weight_others: float, world_size: int, scaler: GradScaler,
    ):
        self.weights = weights
        self.keys: tp.List[str] = list(weights.keys())
        self.weight_others = weight_others

        self.losses: tp.Dict[str, Tensor] = {}
        for key in self.weights.keys():
            self.losses[key] = torch.zeros(1)
        for key in others:
            self.losses[key] = torch.zeros(1)
        self.n_items = 0
        self.world_size = world_size
        self.scaler = scaler
    
    def initialize(self, device='cpu', dtype=torch.float32):
        for key in self.losses.keys():
            self.losses[key] = torch.zeros(1, dtype=dtype, device=device)
        self.n_items = 0
    
    def add_n_items(self, batch_size: int) -> None:
        self.n_items += batch_size
    
    def add_loss(self, losses: tp.Dict[str, Tensor], batch_size: int) -> None:
        for key, loss in losses.items():
            self.losses[key].add_(loss.detach(), alpha=batch_size)
    
    def get(self, key: str) -> float:
        if key not in self.losses:
            return 0.0
        loss = self.losses[key]
        return loss.item() / self.n_items

    def backward(
        self,
        losses: tp.Dict[str, Tensor],
        others: Tensor
    ) -> None:
        loss = others.new_zeros(1)
        for key in self.keys:
            loss += losses[key] * self.weights[key]
        self.scaler.scale(loss + self.weight_others * others).backward()
    
    def reduce_loss(self) -> tp.Dict[str, float]:
        summary = {}
        bucket = torch.cat(list(self.losses.values()))
        dist.reduce(bucket, dst=0, op=dist.ReduceOp.SUM)
        bucket = bucket.cpu()
        n_items = self.n_items * self.world_size
        for idx, key in enumerate(self.losses.keys()):
            summary[f"loss/{key}"] = bucket[idx].item() / n_items
        return summary
    
    def print(self) -> str:
        out = ""
        for name, loss in self.losses.items():
            out = f"{out}  {name}: {loss.item() / self.n_items:.3f}"
        return out
