import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, base_optimizer: Optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer.defaults)
        super(SAM, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.is_first_step = True

    @torch.no_grad()
    def first_step(self, zero_grad=False, set_to_none=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def second_step(self, zero_grad=False, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        if self.is_first_step:
            self.first_step()
            self.is_first_step = False
        else:
            self.second_step()
            self.is_first_step = True

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
