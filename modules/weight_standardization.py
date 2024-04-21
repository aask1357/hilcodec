from typing import TypeVar, Any, Union, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

T_module = TypeVar('T_module', bound=nn.Module)


class WeightStandardization:
    def __init__(
        self,
        name: str,
        dim: Tuple[int],
        eps: float,
        weight: Parameter
    ) -> None:
        self.name = name
        self.eps = eps

        axes = list(range(weight.dim()))
        for d in dim:
            axes.remove(d)
        self.axes = axes

        self.fan_in = 1.0
        for axe in axes:
            self.fan_in *= weight.size(axe)
    
    def compute_weight(self, module: T_module) -> torch.Tensor:
        weight = getattr(module, self.name + "_v")
        gain = getattr(module, self.name + "_g")
        scale = getattr(module, self.name + "_scale")
        
        var, mean = torch.var_mean(weight, dim=self.axes, unbiased=False, keepdim=True)
        weight_standardized = (weight - mean) * torch.rsqrt(torch.clamp(var * self.fan_in, min=self.eps))
        if gain is not None:
            if scale is not None:
                gain = gain * scale
            weight_standardized = gain * weight_standardized
        return weight_standardized
    
    @staticmethod
    def apply(
        module: T_module,
        name: str = "weight",
        dim: Union[int, Tuple[int]] = 0,
        eps: float = 1e-7,
        scale: Optional[float] = None,
        learnable_gain: bool = True,
        zero_init: bool = False
    ) -> 'WeightStandardization':
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightStandardization) and hook.name == name:
                raise RuntimeError("Cannot register two weight_standardize hooks on "
                                   "the same parameter {}".format(name))

        weight = getattr(module, name)
        del module._parameters[name]

        if isinstance(dim, int):
            if dim < -1:
                dim += weight.dim()
            dim = (dim,)

        fn = WeightStandardization(name, dim, eps, weight)

        # Initialize weights with standardized values
        #var, mean = torch.var_mean(weight, dim=fn.axes, unbiased=False, keepdim=True)
        #weight = (weight - mean) * torch.rsqrt(torch.clamp(var * fn.fan_in, min=eps))

        # add g and v as new parameters and express w as g * (w - mean) / (std * sqrt(fan_in))
        # if weight: [c_out, c_in, k_h, k_w] and dim = 0
        # then g: [c_out, 1, 1, 1], fan_in = c_in * k_h * k_w
        module.register_parameter(name + '_v', Parameter(weight.data))
        if learnable_gain:
            g_axes = [1 for _ in range(weight.data.dim())]
            for d in dim:
                g_axes[d] = weight.data.size(d)
            if zero_init:
                g = torch.zeros(*g_axes, dtype=weight.dtype, device=weight.device)
            else:
                g = torch.ones(*g_axes, dtype=weight.dtype, device=weight.device)
            module.register_parameter(name + '_g', Parameter(g.data))
        else:
            module.register_buffer(name + "_g", None)
        
        if scale is not None:
            s = torch.ones(1, dtype=weight.dtype, device=weight.device) * scale
            module.register_buffer(name + "_scale", s)
        else:
            module.register_buffer(name + "_scale", None)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: T_module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_scale']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: T_module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


def weight_standardization(
    module: T_module,
    name: str = "weight",
    dim: Union[int, Tuple[int]] = 0,
    eps: float = 1e-7,
    scale: Optional[float] = None,
    learnable_gain: bool = True,
    zero_init: bool = False
) -> T_module:
    '''Applies weight standardization to a parameter in the given module.
    weight = (weight_gain * scale) * (weight_ - mean(weight_)) / sqrt(var(weight_) * fan_in)

    Args:
        module (nn.Module): Containing module
        name (str, optional): Name of weight parameter to apply weight standardization
        dim (int | Tuple[int], optional): Mean, std, fan_in is calculated except given dim.
        eps (float, optional): Small value to avoid division by zero. Default: 1e-7
        scale (float | None, optional): If not None, scale is multiplied to the weight (not a learnable parameter).
        learnable_gain (bool): Whether to have learnable gain. Default: True
        zero_init (bool): Whether to initialize learnable gain to zero. Default: False
                If learnable_gain == False, the zero_init will be ignored.
    '''
    WeightStandardization.apply(module, name, dim, eps, scale, learnable_gain, zero_init)
    return module

def remove_weight_standardization(module: T_module, name: str = 'weight') -> T_module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightStandardization) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_standardization of '{}' not found in {}"
                     .format(name, module))


if __name__=="__main__":
    m = weight_standardization(nn.Conv1d(2,3,1))
    print(m.weight_.shape, m.weight_gain.shape)
