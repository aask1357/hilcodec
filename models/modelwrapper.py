import os, re
from typing import Optional, Dict, Any
from collections import OrderedDict

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader

from optim import get_optimizer, get_scheduler
from utils import clip_grad_norm_local, plot_param_and_grad


class ModelWrapper_:
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        torch.backends.cudnn.benchmark = True
        self.base_dir: str = hps.base_dir
        self.rank: int = rank
        self.model: torch.nn.Module = self.get_model(hps)
        self.train_mode: bool = train
        self.device = device
        self.epoch: int = 0

        self._module = self.model
        if train:
            hp = hps.train
            self.plot_param_and_grad = hp.plot_param_and_grad
            self.fp16 = hp.fp16
            self.scaler = amp.GradScaler(enabled=hp.fp16)
            torch.cuda.set_device(f'cuda:{rank}')
            self.device = torch.device("cuda", rank)
            if hp.clip_grad is None:
                self.clip_grad = lambda params: None
            elif hp.clip_grad == "norm" or hp.clip_grad == "norm_global":
                self.clip_grad = lambda params: clip_grad_norm_(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "norm_local":
                self.clip_grad = lambda params: clip_grad_norm_local(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "value":
                self.clip_grad = lambda params: clip_grad_value_(params, **hp.clip_grad_kwargs)
            else:
                raise RuntimeError(hp.clip_grad)
            
            self.model.cuda(rank)
            self.optim = get_optimizer(self.model, hp)
            self.scheduler = get_scheduler(self.optim, hp)

            if torch.distributed.get_world_size() > 1:
                self.model = DDP(self.model, device_ids=[rank])
                self._module = self.model.module
        else:
            self.device = device
            self.model.to(device)
    
    def get_model(self, hps) -> torch.nn.Module:
        raise NotImplementedError()
    
    def plot_initial_param(self, dataloader: DataLoader) -> dict:
        hists = {}
        plot_param_and_grad(hists, self.model)
        return hists

    def train_epoch(self, dataloader: DataLoader) -> dict:
        raise NotImplementedError()

    def valid_epoch(self, dataloader: DataLoader) -> dict:
        raise NotImplementedError()
    
    def infer_epoch(self, dataloader: DataLoader) -> dict:
        raise NotImplementedError()
    
    def scheduler_step(self):
        self.sched.step()

    def get_lr(self) -> float:
        return self.optim.param_groups[0]['lr']
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def get_checkpoint(self, epoch: Optional[int]=None, path: Optional[str]=None) -> Dict[str, Any]:
        if path is None:
            if epoch is None:       # get lastest checkpoint
                files = [int(f[:-4]) for f in os.listdir(self.base_dir) if re.match('[0-9]{5,}.pth', f)]
                if not files:
                    if self.rank == 0:
                        print("No checkpoint exists.")
                    return None
                files.sort()
                epoch = files[-1]
            path = os.path.join(self.base_dir, f"{epoch:0>5d}.pth")
        checkpoint = torch.load(path, map_location=self.device)
        if self.rank == 0:
            print(f"Loading checkpoint file '{path}'...")
        return checkpoint
    
    def load(self, epoch: Optional[int] = None, path: Optional[str] = None):
        checkpoint = self.get_checkpoint(epoch, path)
        if checkpoint is None:
            return

        self._module.load_state_dict(checkpoint['model'])
        self.epoch = checkpoint['epoch']
        if self.train_mode:
            self.optim.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            try:
                self.scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                print(e)
        
    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.base_dir, f"{self.epoch:0>5d}.pth")
        torch.save({
            "model": self._module.state_dict(),\
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch
        }, path)
    
    def remove_weight_reparameterizations(self):
        '''remove weight_norm / spectral_norm / weight_standardization / ...'''
        pass


class AudioModelWrapper(ModelWrapper_):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        self.set_textprocessor(hps)
        self.set_keys()
        super().__init__(hps, train, rank, device)
        if self.rank == 0 and self.textprocessor is not None:
            print(f"alphabet size: {self.textprocessor.alphabet_size}")
    
    def set_textprocessor(self, hps):
        if getattr(hps, "text", None) is None:
            self.textprocessor = None
        else:
            from text import TextProcessor
            self.textprocessor = TextProcessor(**hps.text)

    def set_keys(self):
        '''set self.keys, self.infer_keys
        self.keys: used for train_dataset & valid_dataset
        self.infer_keys: used for infer_dataset'''
        self.keys = None
        self.infer_keys = None
