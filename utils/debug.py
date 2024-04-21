from os import path
import torch

class Printer:
    def __init__(self, base_dir):
        self.file = open(path.join(base_dir, "log.txt"), "a")
    
    def __call__(self, s, *args, end='\n'):
        line = "".join([f" {a}" for a in args])
        line = f"{s}{line}"
        print(line, end=end)
        self.file.write(f"{line}{end}")


def check_grad(m, print):
    for name, param in m.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad.data)):
                print(f"****{name}.grad has NaN****")
            elif torch.any(torch.isinf(param.grad.data)):
                print(f"****{name}.grad has +-Inf****")


def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)


def find_not_contributing_params_in_ddp(model, output):
    contributing_parameters = set(get_contributing_params(output))
    for n, p in model.named_parameters():
        if p not in contributing_parameters:
            print(n, p.shape)
