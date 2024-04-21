from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import Dict

import torch
from torchaudio.transforms import Resample
from torch import nn, Tensor
import torch.distributed as dist
from pesq import pesq
from pystoi import stoi
from utils.measure_visqol import measure_visqol as visqol
from utils.terminal import clear_current_line


class Metrics:
    def __init__(self, num_workers: int, sr: int, world_size: int, rank: int, device) -> None:
        self.num_workers = num_workers
        self.sr = sr
        self.world_size = world_size
        self.rank = rank
        self.device = device
        
        if self.sr != 16000:
            self.resample_pesq = Resample(orig_freq=self.sr, new_freq=16000).to(device)
        else:
            self.resample_pesq = nn.Identity()
            self.len_ratio_pesq = 1.0
        if self.sr != 10000:
            self.resample_stoi = Resample(orig_freq=self.sr, new_freq=10000).to(device)
        else:
            self.resample_stoi = nn.Identity()
        
        self.best_pesq: float = 0.0
        self.best_stoi: float = 0.0
        self.best_visqol: float = 0.0
        self.pesq_mean, self.visqol_mean, self.stoi_mean = 0.0, 0.0, 0.0
        self.num_items = 0
        self.pesq_futures, self.visqol_futures = [], []
        self.executor = None
    
    def initialize(self) -> None:
        self.pesq_mean, self.visqol_mean, self.stoi_mean = 0.0, 0.0, 0.0
        self.num_items = 0
        self.pesq_futures, self.visqol_futures = [], []
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
    
    def submit(self, ref: Tensor, deg: Tensor, wav_lens: Tensor) -> None:
        batch_size = ref.size(0)
        deg = deg.to(dtype=torch.float32)
        ref_stoi = self.resample_stoi(ref).cpu().numpy()
        deg_stoi = self.resample_stoi(deg).cpu().numpy()
        ref = self.resample_pesq(ref).cpu().numpy()
        deg = self.resample_pesq(deg).cpu().numpy()

        for i in range(batch_size):
            # PESQ
            wav_len = int(wav_lens[i] * 16000 / self.sr)
            ref_i = ref[i, 0, :wav_len]
            deg_i = deg[i, 0, :wav_len]
            self.pesq_futures.append(self.executor.submit(pesq, 16000, ref_i, deg_i, "wb"))
            
            # ViSQOL
            file_idx = self.world_size * (self.num_items + i) + self.rank
            self.visqol_futures.append(
                self.executor.submit(visqol, ref_i, deg_i, file_idx, "speech")
            )
            
            # STOI
            # On some servers, STOI never finishes if used with multi-processing.
            # Therefore, we use single-processing for STOI.
            wav_len = int(wav_lens[i] * 10000 / self.sr)
            ref_i = ref_stoi[i, 0, :wav_len]
            deg_i = deg_stoi[i, 0, :wav_len]
            self.stoi_mean += stoi(ref_i, deg_i, 10000)
        self.num_items += batch_size
    
    def retrieve(self, verbose: bool) -> Dict[str, float]:
        padding = int(math.log10(self.num_items)) + 1
        for idx, future in enumerate(as_completed(self.pesq_futures), start=1):
            self.pesq_mean += future.result()
            if verbose:
                print(
                    f"\rPESQ {idx:{padding}d}/{self.num_items}"
                    f"    pesq score: {self.pesq_mean / idx:6.4f}",
                    end='', flush=True
                )
        for future in as_completed(self.visqol_futures):
            self.visqol_mean += future.result()
        
        self.executor.shutdown(wait=False, cancel_futures=True)
        
        if verbose:
            clear_current_line()
        
        if self.world_size > 0:
            metric_mean = torch.tensor(
                [self.pesq_mean, self.visqol_mean, self.stoi_mean],
                device=self.device
            )
            dist.reduce(metric_mean, dst=0, op=dist.ReduceOp.SUM)
            self.pesq_mean = metric_mean[0].item()
            self.visqol_mean = metric_mean[1].item()
            self.stoi_mean = metric_mean[2].item()
            self.num_items *= self.world_size
        
        self.pesq_mean /= self.num_items
        self.stoi_mean /= self.num_items
        self.visqol_mean /= self.num_items
        if self.best_pesq < self.pesq_mean:
            self.best_pesq = self.pesq_mean
        if self.best_visqol < self.visqol_mean:
            self.best_visqol = self.visqol_mean
        if self.best_stoi < self.stoi_mean:
            self.best_stoi = self.stoi_mean
        
        return {
            "metrics/pesq": self.pesq_mean,
            "metrics/visqol": self.visqol_mean,
            "metrics/stoi": self.stoi_mean,
            "metrics/best_pesq": self.best_pesq,
            "metrics/best_visqol": self.best_visqol,
            "metrics/best_stoi": self.best_stoi,
        }
    
    def state_dict(self) -> Dict[str, float]:
        return {
            "best_pesq": self.best_pesq,
            "best_visqol": self.best_visqol,
            "best_stoi": self.best_stoi,
        }
    
    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.best_pesq = state_dict["best_pesq"]
        self.best_visqol = state_dict["best_visqol"]
        self.best_stoi = state_dict["best_stoi"]
