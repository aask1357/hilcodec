import typing as tp
import warnings
import torch
from torch import Tensor
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


DiscOutput = tp.Dict[str, tp.List[Tensor]]
LossOutput = tp.Dict[str, Tensor]


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class MelLoss(torch.nn.Module):
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
        y_pred = mel_spectrogram(
            y_pred, self.mel_basis, self.window, self.n_fft, self.hop_size,
            self.win_size, self.center
        )
        y_true = mel_spectrogram(
            y_true, self.mel_basis, self.window, self.n_fft, self.hop_size,
            self.win_size, self.center
        )
        return {"freq": F.l1_loss(y_pred, y_true)}


def mel_spectrogram(y, mel_basis, window, n_fft, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    y = torch.nn.functional.pad(
        y,
        (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)),
        mode='reflect'
    ).squeeze(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = torch.stft(
            y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
            window=window, center=center,
            pad_mode='reflect', normalized=False, onesided=True
        )

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

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
    return loss #/ n_logits


def generator_loss_lsgan(logits: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in logits.keys():
        loss_n = 0.0
        for dg in logits[name]:
            loss_n += ((1.0 - dg).square()).mean()
        loss[f"{name}_g"] = loss_n #/ len(logits[name])
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
    return loss #/ n_logits


def generator_loss(logits: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in logits.keys():
        loss_n = 0.0
        for lg in logits[name]:
            loss_n += F.relu(1 - lg).mean()
        loss[f"{name}_g"] = loss_n #/ len(logits[name])
    return loss


def feature_loss(fmaps_g: DiscOutput, fmaps_r: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in fmaps_g.keys():
        loss_n = 0.0
        for g, r in zip(fmaps_g[name], fmaps_r[name]):
            r = r.detach()
            loss_n += F.l1_loss(g, r)
        loss[f"{name}_fm"] = loss_n #/ len(fmaps_g[name])
    return loss


def feature_loss_normalized(fmaps_g: DiscOutput, fmaps_r: DiscOutput) -> LossOutput:
    loss: LossOutput = {}
    for name in fmaps_g.keys():
        loss_n = 0.0
        for g, r in zip(fmaps_g[name], fmaps_r[name]):
            r = r.detach()
            loss_n += F.l1_loss(g, r) / r.abs().mean().clamp_min(1e-12)
        loss[f"{name}_fm"] = loss_n #/ len(fmaps_g[name])
    return loss
