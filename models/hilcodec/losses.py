import typing as tp
import warnings
import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from librosa.filters import mel as librosa_mel_fn


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
        loss = wav_g.new_zeros(1)
        for transform in self.mel_transforms.values():
            mel_g = transform(wav_g)
            mel_g = torch.where(
                mel_g >= self.clip_val,
                mel_g,
                mel_g - mel_g.detach() + self.clip_val
            ).log_()
            mel_r = transform(wav_r).clamp_min_(self.clip_val).log_()
            loss = loss + F.mse_loss(mel_g, mel_r) + F.l1_loss(mel_g, mel_r) 
        loss_dict = {"freq": loss}
        return loss_dict


class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad: torch.Tensor, loss: torch.Tensor):
        ctx._grad = grad
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx._grad * grad_output
        return grad, None, None


class CustomMelSpectrogram(nn.Module):
    def __init__(self, sr, n_fft, hop_size, n_mels, norm):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels,
                             fmin=0.0, fmax=None, norm=norm)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(n_fft))
        self.mel_basis: Tensor
        self.window: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        return mel_spectrogram(x, self.mel_basis, self.window, self.n_fft,
                               self.hop_size, win_size=self.n_fft, center=False)


class MelGradFunction(nn.Module):
    def __init__(self, sampling_rate: int, clip_val: float,
                 n_mels_max: int, mel_norm: tp.Optional[str] = None):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.clip_val = clip_val
        
        self.mel_transforms = nn.ModuleDict()
        for i in range(5, 11):
            s = 2**i
            # Make sure that every filter has at least one non-zero value
            n_mels = int(min(
                n_mels_max,
                2*mel_scale(sampling_rate/2) / mel_scale(sampling_rate/s) - 1,
                s//4
            ))
            self.mel_transforms[f"freq{s}"] = CustomMelSpectrogram(
                sr=sampling_rate, n_fft=s, hop_size=s//4,
                n_mels=n_mels, norm=mel_norm)

    def forward(self, wav_g: Tensor, wav_r: Tensor) -> LossOutput:
        loss = wav_g.new_zeros(1)
        for transform in self.mel_transforms.values():
            mel_g = transform(wav_g)
            mel_r = transform(wav_r)
            with torch.no_grad():
                log_mel_g = mel_g.clamp_min_(self.clip_val).log_()
                log_mel_r = mel_r.clamp_min_(self.clip_val).log_()
                _loss = F.l1_loss(log_mel_g, log_mel_r) + F.mse_loss(log_mel_g, log_mel_r)
                _grad = (log_mel_g - log_mel_r) / log_mel_r.numel()
            loss += GradFunction.apply(mel_g, _grad, _loss)
        loss_dict = {"freq": loss}
        return loss_dict


def log_clip(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class HifiGANMelLoss(nn.Module):
    def __init__(self, sampling_rate, clip_val, n_fft, num_mels,
                 hop_size, win_size, fmin=0, fmax=None, center=False):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.clip_val = clip_val
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(win_size))
        self.mel_basis: Tensor
        self.window: Tensor

    def forward(self, y_pred, y_true):
        y_pred = log_clip(mel_spectrogram(
            y_pred, self.mel_basis, self.window, self.n_fft, self.hop_size,
            self.win_size, self.center
        ))
        y_true = log_clip(mel_spectrogram(
            y_true, self.mel_basis, self.window, self.n_fft, self.hop_size,
            self.win_size, self.center
        ))
        return {"freq": F.l1_loss(y_pred, y_true)}


def mel_spectrogram(y, mel_basis, window, n_fft, hop_size, win_size, center=False):
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y, (padding, padding), mode='reflect').squeeze(1)
    spec = torch.stft(
        y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
        window=window, center=center,
        pad_mode='reflect', normalized=False, onesided=True,
        return_complex=True
    ).abs()
    spec = torch.matmul(mel_basis, spec)
    return spec


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
