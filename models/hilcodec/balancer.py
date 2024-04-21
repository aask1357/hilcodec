import typing as tp

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch import autograd
from torch import distributed as dist


class Balancer:
    """Loss balancer.
    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.
    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)
    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.
    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        scaled: Loss which is scaled by a GradScaler
    """

    def __init__(
        self, weights: tp.Dict[str, float], others: tp.List[str],
        weight_others: float, world_size: int,
        scaler_g: GradScaler, scaler_d: GradScaler,
        ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.epsilon = epsilon
        self.ema_decay = ema_decay
        self.keys: tp.List[str] = list(weights.keys())
        self.ema_norms: tp.Dict[str, Tensor] = {}
        self.ema_fix: float = 0.0
        self._init_ema_norms: tp.Dict[str, float] = {}
        self.weight_others = weight_others

        self.losses: tp.Dict[str, Tensor] = {}
        for key in self.weights.keys():
            self.losses[key] = torch.zeros(1)
        for key in others:
            self.losses[key] = torch.zeros(1)
        self.n_items = 0
        self.world_size = world_size
        self.scaler_g = scaler_g
        self.scaler_d = scaler_d
    
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
        input: Tensor,
        others: Tensor
    ) -> bool:
        norms: tp.Dict[str, Tensor] = {}
        grads: tp.Dict[str, Tensor] = {}
        bucket = torch.empty(len(self.keys), dtype=torch.float32, device=input.device)
        
        if self.scaler_d._enabled:
            scale = self.scaler_d._scale
            invscale = scale.double().reciprocal().float()
        else:
            scale = 1.0
            invscale = 1.0

        # Lazy initialization (since we don't know the device before the first call)
        if not self.ema_norms:
            if self._init_ema_norms:
                for key, value in self._init_ema_norms.items():
                    if key == "fix":
                        self.ema_fix = value
                        continue
                    self.ema_norms[key] = torch.tensor(
                        [value], dtype=torch.float32, device=input.device)
            else:
                for key in self.keys:
                    self.ema_norms[key] = torch.tensor(
                        [0.0], dtype=torch.float32, device=input.device)

        for idx, key in enumerate(self.keys):
            loss = losses[key]
            
            # loss_g & loss_fm -> scaled by scaler_d
            if key.endswith("_g") or key.endswith("_fm"):
                loss = loss * scale
                _invscale = invscale
            else:
                _invscale = 1.0
            grad, = autograd.grad(loss, [input], retain_graph=True)
            grad = grad.float() * _invscale
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[key] = norm
            grads[key] = grad

            norm_prev = self.ema_norms[key]
            ema_norm = self.ema_decay * norm_prev + (1 - self.ema_decay) * norm
            bucket[idx] = ema_norm
        bucket /= self.world_size
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
        self.ema_fix = self.ema_fix * self.ema_decay + (1.0 - self.ema_decay)

        if not torch.all(torch.isfinite(bucket)):
            self.scaler_d.update(scale * self.scaler_d._backoff_factor)
            # Even if we don't have to perform backward theoretically,
            # we still need to perform backward to release the memory.
            # If we don't, then the memory usage will increase.
            (input*0.0).sum().backward()
            return False

        out_grad: tp.Union[float, Tensor] = 0.0
        ema_reciprocal = (bucket / self.ema_fix + self.epsilon).reciprocal()
        for idx, key in enumerate(self.keys):
            self.ema_norms[key] = bucket[idx]
            scale = self.weights[key] * ema_reciprocal[idx]
            out_grad += scale * grads[key]

        loss: Tensor = (input * out_grad.detach()).sum() + others * self.weight_others
        self.scaler_g.scale(loss).backward()
        return True
    
    def reduce_loss(self) -> tp.Dict[str, float]:
        summary = {}
        bucket = torch.cat(list(self.losses.values()))
        dist.reduce(bucket, dst=0, op=dist.ReduceOp.SUM)
        bucket = bucket.cpu()
        n_items = self.n_items * self.world_size
        for idx, key in enumerate(self.losses.keys()):
            summary[f"loss/{key}"] = bucket[idx].item() / n_items
        return summary

    def reduce_ema(self) -> tp.Dict[str, float]:
        # ema_norms are already reduced during self.backward()
        summary = {}
        for name, norm in self.ema_norms.items():
            summary[f"ema_norm/{name}"] = norm.item() / self.ema_fix
        return summary
    
    def print(self, print_ema: bool = False) -> str:
        out = ""
        if print_ema:
            for name, loss in self.losses.items():
                out = f"{out}  loss/{name}: {loss.item() / self.n_items:.3f}"
            for name, value in self.ema_norms.items():
                out = f"{out}  ema/{name}: {value.item():.3e}"
        else:
            for name, loss in self.losses.items():
                out = f"{out}  {name}: {loss.item() / self.n_items:.3f}"
        return out

    def state_dict(self) -> tp.Dict[str, float]:
        state_dict = {key: value.item() for key, value in self.ema_norms.items()}
        state_dict["fix"] = self.ema_fix
        return state_dict

    def load_state_dict(self, state_dict: tp.Dict[str, float]):
        if len(state_dict) == 0:
            return
        if self.ema_norms:
            for key in state_dict.keys():
                if key == "fix":
                    self.ema_fix: float = state_dict[key]
                    continue
                self.ema_norms[key].fill_(state_dict[key])
        else:
            self._init_ema_norms = {key: value for key, value in state_dict.items()}
