import typing as tp
import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


DiscOutput = tp.Dict[str, tp.List[Tensor]]
LossOutput = tp.Dict[str, Tensor]


def mel_scale(f: tp.Union[int, float]) -> float:
    return 2595 * math.log10(1 + f / 700)


class MelLoss(nn.Module):
    def __init__(self, sampling_rate: int, clip_val: float = 1.0e-5,
                 no_zero: bool = True, n_mels_max: int = 80):
        super().__init__()
        self.clip_val = clip_val
        self.mel_transforms = nn.ModuleDict()
        for i in range(5, 11):
            s = 2**i
            if no_zero:
                # Make sure that every filter has at least one non-zero value
                n_mels = int(min(
                    n_mels_max,
                    2*mel_scale(sampling_rate/2) / mel_scale(sampling_rate/s) - 1,
                    s//4
                ))
            else:
                n_mels = min(n_mels_max, s//4)
            self.mel_transforms[f"freq{s}"] = MelSpectrogram(
                sample_rate=sampling_rate, n_fft=s,
                win_length=s, hop_length=s//4, n_mels=n_mels, normalized=False,
                center=False, norm='slaney')
    
    def forward(self, wav_g: Tensor, wav_r: Tensor) -> LossOutput:
        loss: LossOutput = {}
        for name, transform in self.mel_transforms.items():
            mel_g = transform(wav_g)
            mel_g = torch.where(
                mel_g >= self.clip_val,
                mel_g,
                mel_g - mel_g.detach() + self.clip_val
            ).log()
            mel_r = transform(wav_r).clamp_min(self.clip_val).log()
            loss[name] = F.mse_loss(mel_g, mel_r) + F.l1_loss(mel_g, mel_r)
        return loss


def discriminator_loss_lsgan(logits_g: DiscOutput, logits_r: DiscOutput) -> Tensor:
    loss = 0.0
    n_logits = 0
    for name in logits_g.keys():
        for dg, dr in zip(logits_g[name], logits_r[name]):
            r_loss = ((1.0 - dr).square()).mean()
            g_loss = (dg.square()).mean()
            loss += (r_loss + g_loss)
            n_logits += 1
    return loss / n_logits


def generator_loss_lsgan(logits: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in logits.keys():
        loss_n = 0.0
        for dg in logits[name]:
            loss_n += ((1.0 - dg).square()).mean()
        loss[f"{name}_g"] = loss_n / len(logits[name])
    return loss


def discriminator_loss(logits_g: DiscOutput, logits_r: DiscOutput) -> Tensor:
    loss = 0.0
    n_logits = 0
    for name in logits_g.keys():
        for lg, lr in zip(logits_g[name], logits_r[name]):
            r_loss = F.relu(1 - lr).mean()
            g_loss = F.relu(1 + lg).mean()
            loss += (r_loss + g_loss)
            n_logits += 1
    return loss / n_logits


def generator_loss(logits: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in logits.keys():
        loss_n = 0.0
        for lg in logits[name]:
            loss_n += F.relu(1 - lg).mean()
        loss[f"{name}_g"] = loss_n / len(logits[name])
    return loss


def feature_loss(fmaps_g: DiscOutput, fmaps_r: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in fmaps_g.keys():
        loss_n = 0.0
        for g, r in zip(fmaps_g[name], fmaps_r[name]):
            r = r.detach()
            loss_n += F.l1_loss(g, r)
        loss[f"{name}_fm"] = loss_n / len(fmaps_g[name])
    return loss


def feature_loss_normalized(fmaps_g: DiscOutput, fmaps_r: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in fmaps_g.keys():
        loss_n = 0.0
        for g, r in zip(fmaps_g[name], fmaps_r[name]):
            r = r.detach()
            loss_n += F.l1_loss(g, r) / r.abs().mean().clamp_min(1e-12)
        loss[f"{name}_fm"] = loss_n / len(fmaps_g[name])
    return loss
