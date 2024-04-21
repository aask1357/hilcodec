import os
import math
from typing import Optional, Dict

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

from models.modelwrapper import AudioModelWrapper
from functional import mel_spectrogram
from optim import get_optimizer, get_scheduler
from utils import plot_param_and_grad, clip_grad_norm_local
from utils.data import get_dataset_dataloader
from utils.terminal import clear_current_line

from .models import EncodecModel
from .discriminators import Discriminators
from .losses import feature_loss, generator_loss, discriminator_loss, MelLoss, \
    feature_loss_normalized, generator_loss_lsgan, discriminator_loss_lsgan
from .balancer import Balancer
from .metrics import Metrics


class ModelWrapper(AudioModelWrapper):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        self.base_dir = hps.base_dir
        self.rank = rank
        self.textprocessor = None
        self.h = hps.data

        self.model = EncodecModel(sample_rate=hps.data.sampling_rate, 
            channels_audio=hps.data.channels, **hps.model_kwargs)
        self.disc = Discriminators(**hps.disc_kwargs)
        self._module = self.model
        self._disc = self.disc
        self.num_quantizers: int = hps.model_kwargs.vq_kwargs.num_quantizers
        self.lookahead: int = getattr(hps.train, "lookahead", 0)

        self.hop_size = 1
        for n in hps.model_kwargs.strides:
            self.hop_size *= n
        self.mel_loss = MelLoss(
            hps.data.sampling_rate, hps.data.clip_val,
            no_zero=getattr(hps.train, "no_zero_at_mel_filter", True),
            n_mels_max=getattr(hps.train, "n_mels_max", 80)
        )

        self.train_mode = train
        self.epoch = 0
        self.keys = ["wav"]
        self.infer_keys = ["wav"]

        if not train:
            self.device = device
            self.model.to(device)
            self.mel_loss.to(device)
        else:
            hp = hps.train
            self.disc_update_ratio = getattr(hp, "disc_update_ratio", [1, 1])
            self.infer_n: Optional[int] = getattr(hp, "infer_n", None)
            self.device = torch.device("cuda", rank)

            self.fp16_g = hp.fp16_g
            self.fp16_d = hp.fp16_d
            self.plot_param_and_grad = hp.plot_param_and_grad
            self.world_size = dist.get_world_size()
            
            if hp.clip_grad is None:
                self.clip_grad = lambda params: None
            elif hp.clip_grad == "norm" or hp.clip_grad == "norm_global":
                self.clip_grad = lambda params: clip_grad_norm_(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "norm_local":
                self.clip_grad = lambda params: clip_grad_norm_local(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "value":
                self.clip_grad = lambda params: clip_grad_value_(params, **hp.clip_grad_kwargs)
            
            self.mel_loss.cuda(rank)
            self.model.cuda(rank)
            self.disc.cuda(rank)
            self.optim_g = get_optimizer(self.model, hp)
            self.optim_d = get_optimizer(self.disc, hp)
            self.scheduler_g = get_scheduler(self.optim_g, hp)
            self.scheduler_d = get_scheduler(self.optim_d, hp)
            
            self.initialized = False
            if self.world_size > 1:
                self.model = DDP(self.model, device_ids=[rank], broadcast_buffers=False,
                                 gradient_as_bucket_view=True,
                                 static_graph=True)
                self.disc = DDP(self.disc, device_ids=[rank], broadcast_buffers=False,
                                gradient_as_bucket_view=True,
                                static_graph=True)
                self._module = self.model.module
                self._disc = self.disc.module
            self.scaler_g = amp.GradScaler(init_scale=2.0**3, enabled=self.fp16_g)
            self.scaler_d = amp.GradScaler(init_scale=2.0**15, enabled=self.fp16_d)
            self.disc_parameters = list(self.disc.parameters())
            
            self.balancer = Balancer(others=["d", "vq"], world_size=self.world_size,
                scaler_d=self.scaler_d, scaler_g=self.scaler_g, **hp.balancer_kwargs)

            if not getattr(hps.train, "deterministic", False):
                torch.backends.cudnn.benchmark = True
            
            self.pesq_interval: int = hps.pesq.interval
            _, self.pesq_loader = get_dataset_dataloader(
                hps, mode="pesq", keys=["wav", "wav_len"], n_gpus=self.world_size, rank=rank)
            self.metrics = Metrics(
                num_workers=hps.pesq.num_workers_executor, sr=hps.data.sampling_rate,
                world_size=self.world_size, rank=rank, device=self.device)

            if getattr(hp, "use_lsgan", True):
                self.d_loss_fn = discriminator_loss_lsgan
                self.g_loss_fn = generator_loss_lsgan
            else:
                self.d_loss_fn = discriminator_loss
                self.g_loss_fn = generator_loss

            if getattr(hp, "use_normalized_fm_loss", False):
                self.fm_loss_fn = feature_loss_normalized
            else:
                self.fm_loss_fn = feature_loss

    def plot_initial_param(self, dataloader: DataLoader) -> dict:
        hists = {}
        plot_param_and_grad(hists, self._disc, "disc")
        plot_param_and_grad(hists, self._module, "model")
        return hists

    def set_d_requires_grad(self, requires_grad: bool):
        for param in self.disc.parameters():
            param.requires_grad = requires_grad
    
    def initialize(self, dataloader):
        # 1. For VQ initialization. If we don't use separate initialization func,
        #    the GPU memory used for the init will not be released until the end of
        #    the first iteration.
        # 2. For DDP. DDP(gradient_as_bucket_view=True) use less GPU memory
        #    after the first iteration.
        # 3. Initialize scaler_d. torch.cuda.amp.GradScaler use lazy initialization
        #    so scaler_d._scale is not initialized at the beginning,
        #    which is needed for balancer.
        if self.rank == 0:
            print("Initializing model...")

        # scaler_d._scale initialization
        self.scaler_d.scale(torch.zeros(1, device=self.device).mean())

        for batch in dataloader:
            wav_r = batch["wav"].cuda(self.rank, non_blocking=True).unsqueeze(1)
            # VQ initialization
            with amp.autocast(enabled=self.fp16_g):
                wav_g, _, loss_vq = self.model(wav_r, n=self.num_quantizers)
            
            # DDP gradient_bucket initialization
            if self.world_size > 1:
                # multipy 0 to avoid unstable gradient when using fp16
                ((wav_g + loss_vq) * 0.0).sum().backward()

                tmp = torch.zeros_like(wav_r[0:1])  # use batch_size 1 for discriminator.
                with amp.autocast(enabled=self.fp16_d):
                    logits, _ = self.disc(tmp)
                loss = 0.0
                for logit in logits.values():
                    for l in logit:
                        loss += (l * 0.0).sum()
                loss.backward()
            break

        if self.rank == 0:
            print("Done.")

    def train_epoch(self, dataloader):
        self.train()
        if not self.initialized:
            self.initialize(dataloader)
            self.initialized = True
        
        self.balancer.initialize(device=self.device, dtype=torch.float32)
        max_items = len(dataloader)
        padding = int(math.log10(max_items)) + 1
        num_replaces_total = None
        
        loss_d = torch.zeros(1, device=self.device)
        summary = {"scalars": {}, "hists": {}, "scalars_not_to_print": {}}
        # self.set_d_requires_grad(True)
        for idx, batch in enumerate(dataloader, start=1):
            wav_r = batch["wav"].cuda(self.rank, non_blocking=True).unsqueeze(1)
            batch_size = wav_r.size(0)
            self.balancer.add_n_items(batch_size)
            
            with amp.autocast(enabled=self.fp16_g):
                wav_g, num_replaces, loss_vq = self.model(wav_r)
                if self.lookahead > 0:
                    wav_r = wav_r[:, :, :-self.lookahead]
                    wav_g = wav_g[:, :, self.lookahead:]
            
            with amp.autocast(enabled=self.fp16_d):
                logits_g, fmaps_g = self.disc(wav_g)
                logits_r, fmaps_r = self.disc(wav_r)

            # update Generator
            # self.set_d_requires_grad(False)
            self.optim_g.zero_grad()
            loss_dict: Dict[str, torch.Tensor] = self.mel_loss(wav_g, wav_r)
            with amp.autocast(enabled=self.fp16_d):
                loss_dict.update(self.fm_loss_fn(fmaps_g, fmaps_r))
                loss_dict.update(self.g_loss_fn(logits_g))
            self.balancer.add_loss(loss_dict, batch_size)
            
            # if self.world_size > 1:
            #     with self.disc.no_sync():
            #         successful = self.balancer.backward(loss_dict, wav_g, loss_vq)
            # else:
            successful = self.balancer.backward(loss_dict, wav_g, loss_vq)
            if successful:
                self.scaler_g.unscale_(self.optim_g)
                if idx == len(dataloader) and self.rank == 0 and self.plot_param_and_grad:
                    plot_param_and_grad(summary["hists"], self._module, "model")
                self.clip_grad(self.model.parameters())
                self.scaler_g.step(self.optim_g)
                self.scaler_g.update()
                if num_replaces_total is None:
                    num_replaces_total = num_replaces
                else:
                    num_replaces_total += num_replaces
            # self.set_d_requires_grad(True)
            
            # update Discriminator
            r0, r1 = self.disc_update_ratio[0], self.disc_update_ratio[1]
            if idx % r1 < r0:
                self.optim_d.zero_grad()
                with amp.autocast(enabled=self.fp16_d):
                    loss_d = self.d_loss_fn(logits_g, logits_r)

                self.scaler_d.scale(loss_d).backward(inputs=self.disc_parameters)
                self.scaler_d.unscale_(self.optim_d)
                if idx == len(dataloader) // r1 * r1 and self.plot_param_and_grad:
                    plot_param_and_grad(summary["hists"], self._disc, "disc")
                self.clip_grad(self.disc_parameters)
                self.scaler_d.step(self.optim_d)
                self.scaler_d.update()
            self.balancer.add_loss({"vq": loss_vq, "d": loss_d}, batch_size)

            if self.rank == 0:
                print(
                    f"\rEpoch {self.epoch} - "
                    f"Train {idx:{padding}d}/{max_items} ({idx/max_items*100:>4.1f}%)"
                    f"{self.balancer.print()}"
                    f"  scale_g {self.scaler_g.get_scale():.3e}"
                    f"  scale_d {self.scaler_d.get_scale():.3e}"
                    f"  lr: {self.get_lr():.2e}",
                    end='', flush=True
                )
            if hasattr(self.scheduler_g, "warmup_step"):
                self.scheduler_g.warmup_step()
                self.scheduler_d.warmup_step()
        # ensure that buffers are consistent across all ranks
        # if self.world_size > 1:
        #     if self.rank == 0:
        #         clear_current_line()
        #         print("\rChecking for buffer consistency...", end='', flush=True)
        #     for vq_layer in self._module.quantizer.layers:
        #         codebook = vq_layer._codebook
        #         for name in ["embed", "ema_embed", "ema_num"]:
        #             buf: torch.Tensor = getattr(codebook, name)
        #             bucket = [torch.empty_like(buf) for _ in range(self.world_size)]
        #             dist.all_gather(bucket, buf.clone())
        #             for buf_from_other_rank in bucket:
        #                 assert torch.allclose(buf, buf_from_other_rank)
        if self.world_size > 1:
            if self.rank == 0:
                clear_current_line()
                print("\rChecking for Discriminators consistency...", end='', flush=True)
            for n, p in self._disc.named_parameters():
                bucket = [torch.empty_like(p.data) for _ in range(self.world_size)]
                dist.all_gather(bucket, p.data.clone())
                for buf_from_other_rank in bucket:
                    assert torch.allclose(p.data, buf_from_other_rank), n

        if self.rank == 0:
            clear_current_line()
        self.scheduler_g.step()
        self.scheduler_d.step()

        summary["scalars"] = self.balancer.reduce_loss()
        summary["scalars_not_to_print"] = self.balancer.reduce_ema()
        for idx in range(len(num_replaces_total)):
            summary["scalars_not_to_print"][f"n_replaces/{idx}"] = num_replaces_total[idx]
        summary["scalars"]["scale/g"] = self.scaler_g.get_scale()
        summary["scalars"]["scale/d"] = self.scaler_d.get_scale()
        summary["scalars"]["lr"] = self.get_lr()
        return summary
    
    @torch.no_grad()
    def valid_epoch(self, dataloader):
        self.eval()
        summary = self._valid_epoch(dataloader)
        if self.epoch % self.pesq_interval == 0:
            metrics = self.pesq_epoch()
            summary["scalars"].update(metrics)
        return summary

    def _valid_epoch(self, dataloader):
        self.balancer.initialize(device=torch.device("cuda", index=self.rank), dtype=torch.float32)
        for batch in tqdm(dataloader, desc="Valid", disable=(self.rank!=0), leave=False, dynamic_ncols=True):
            wav_r = batch["wav"].cuda(self.rank, non_blocking=True).unsqueeze(1)
            batch_size = wav_r.size(0)
            self.balancer.add_n_items(batch_size)

            with amp.autocast(enabled=self.fp16_g):
                wav_g, _, loss_vq = self._module(wav_r, n=self.infer_n)
                if self.lookahead > 0:
                    wav_r = wav_r[:, :, :-self.lookahead]
                    wav_g = wav_g[:, :, self.lookahead:]

            loss_dict = {'vq': loss_vq}
            loss_dict.update(self.mel_loss(wav_g, wav_r))
            with amp.autocast(enabled=self.fp16_d):
                logits_g, fmaps_g = self.disc(wav_g)
                logits_r, fmaps_r = self.disc(wav_r)
                loss_dict["d"] = self.d_loss_fn(logits_g, logits_r)
                loss_dict.update(self.g_loss_fn(logits_g))
                loss_dict.update(self.fm_loss_fn(fmaps_g, fmaps_r))
            self.balancer.add_loss(loss_dict, batch_size)
        summary_scalars = self.balancer.reduce_loss()
        return {"scalars": summary_scalars}
    
    def pesq_epoch(self) -> Dict[str, float]:
        self.metrics.initialize()
        for batch in tqdm(self.pesq_loader, desc="PESQ", disable=(self.rank!=0), leave=False, dynamic_ncols=True):
            wav_r = batch["wav"].cuda(self.rank, non_blocking=True).unsqueeze(1)
            wav_len = batch["wav_len"]

            batch_wav_len = wav_r.size(-1) // self.hop_size * self.hop_size
            wav_r = wav_r[:, :, :batch_wav_len]
            with amp.autocast(enabled=self.fp16_g):
                wav_g, *_ = self._module(wav_r, n=self.infer_n)
                if self.lookahead > 0:
                    wav_r = wav_r[:, :, :-self.lookahead]
                    wav_g = wav_g[:, :, self.lookahead:]
            self.metrics.submit(wav_r, wav_g, wav_len)
        
        metrics = self.metrics.retrieve(verbose=(self.rank==0))
        return metrics

    @torch.no_grad()
    def infer_epoch(self, dataloader):
        self.eval()
        summary = {"audios": {}, "specs": {}}
        for iter, batch in enumerate(dataloader, start=1):
            wav_r = batch["wav"].cuda(self.rank).unsqueeze(1)
            wav_len = wav_r.size(2) // self.hop_size * self.hop_size
            wav_r = wav_r[:, :, :wav_len]
            with torch.no_grad():
                wav_g, *_ = self._module(wav_r)
                if self.lookahead > 0:
                    wav_r = wav_r[:, :, :-self.lookahead]
                    wav_g = wav_g[:, :, self.lookahead:]
                mel_g = mel_spectrogram(wav_g.squeeze(1), self.h.n_fft, 80,
                    self.h.sampling_rate, self.h.hop_size, self.h.win_size)
            summary["audios"][f"gen/wav_{iter}"] = wav_g.squeeze().cpu().numpy()
            summary["specs"][f"gen/mel_{iter}"] = mel_g.squeeze().cpu().numpy()

            if self.epoch == 1:
                summary["audios"][f"gt/wav_{iter}"] = wav_r.squeeze().cpu().numpy()
                mel_r = mel_spectrogram(wav_r.squeeze(1), self.h.n_fft, 80,
                    self.h.sampling_rate, self.h.hop_size, self.h.win_size)
                summary["specs"][f"gt/mel_{iter}"] = mel_r.squeeze().cpu().numpy()
        return summary

    def load(self, epoch: Optional[int] = None, path: Optional[str] = None):
        checkpoint = self.get_checkpoint(epoch, path)
        if checkpoint is None:
            return

        self._module.load_state_dict(checkpoint['model'])
        self._disc.load_state_dict(checkpoint['disc'])
        self.epoch = checkpoint['epoch']

        if self.train_mode:
            self.optim_g.load_state_dict(checkpoint['optim_g'])
            self.optim_d.load_state_dict(checkpoint['optim_d'])
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
            self.scaler_g.load_state_dict(checkpoint['scaler_g'])
            self.scaler_d.load_state_dict(checkpoint['scaler_d'])
            self.balancer.load_state_dict(checkpoint['balancer'])
            self.metrics.load_state_dict(checkpoint['metrics'])
    
    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.base_dir, f"{self.epoch:0>5d}.pth")
        wrapper_dict = {
            "model": self._module.state_dict(),
            "disc": self._disc.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "epoch": self.epoch,
            "balancer": self.balancer.state_dict(),
            "metrics": self.metrics.state_dict(),
        }
        torch.save(wrapper_dict, path)
    
    def train(self):
        self.model.train()
        self.disc.train()
    
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self.disc.to(*args, **kwargs)
        self.mel_loss.to(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        self.disc.eval()

    def get_lr(self):
        return self.optim_g.param_groups[0]['lr']
    