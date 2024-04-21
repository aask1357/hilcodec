import os
import re
import math
import wave
import random
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.interpolate import interp1d
from tqdm import tqdm
try:
    import parselmouth  # used for pitch extractor
except ImportError:
    pass

from functional import stft, spec_to_mel


class TensorDict(dict):
    '''To set pin_memory=True in DataLoader, need to use this class instead of dict.'''
    def pin_memory(self):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.pin_memory()


class NormalizeMethods(Enum):
    NULL = 0
    MAX = 1
    RANDOMGAIN = 2


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys, textprocessor=None, mode="train", batch_size=1, verbose=True):
        super().__init__()
        self.wav_dir = hp.wav_dir
        self.data_dir = hp.data_dir
        self.segment_size: Optional[int]
        if mode in ["infer", "pesq"]:
            self.segment_size = None
        else:
            self.segment_size = getattr(hp, "segment_size", None)
        _filter = hp.filter.get(mode, False)
        
        self.textprocessor = textprocessor
        self.keys = keys
        self.hp = hp

        self.normalize_method: NormalizeMethods
        self.random_gain_low: float = 1.0
        self.random_gain_high: float = 1.0
        method = getattr(hp, "normalize_method", "max")
        if method == "max":
            self.normalize_method = NormalizeMethods.MAX
        elif method == "null" or method is None:
            self.normalize_method = NormalizeMethods.NULL
        elif method == "random_gain":
            if mode == "train":
                self.normalize_method = NormalizeMethods.RANDOMGAIN
                self.random_gain_low: float = hp.random_gain_low
                self.random_gain_high: float = hp.random_gain_high
            else:
                self.normalize_method = NormalizeMethods.NULL
        else:
            raise RuntimeError(f"hps.data.normalize_method {method} is not supported.")

        filelist = hp.filelists[mode]

        self.wav_idx, self.text_idx = [], []
        with open(filelist, encoding="utf-8") as txt:
            wav_text_tag = [l.strip().split("|") for l in txt.readlines()]
        if mode=="infer":
            wav_text_tag = wav_text_tag[:hp.num_infer]
        for wtt in wav_text_tag:
            self.wav_idx.append(re.sub(f"\.{hp.extension}$", "", wtt[0]))
            self.text_idx.append(wtt[1:])

        if _filter:
            self.batch_size = batch_size
            # 1. filter out very short or long utterances
            wav_idx_ = []
            text_idx_ = []
            wav_len = []
            min_length = getattr(hp, "min_length", 0)
            max_length = getattr(hp, "max_length", float("inf"))
            for i in tqdm(range(len(self.wav_idx)), desc=f"Filtering {mode} dataset", dynamic_ncols=True, leave=False, disable=(not verbose)):
                wav_length = self.get_wav_length(i)
                if min_length < wav_length < max_length:
                    wav_idx_.append(self.wav_idx[i])
                    text_idx_.append(self.text_idx[i])
                    wav_len.append(wav_length)
            if verbose:
                print(f'{mode} dataset filtered: {len(wav_idx_)}/{len(self.wav_idx)}')

            # 2. group wavs with similar lengths in a same batch
            idx_ascending = np.array(wav_len).argsort()
            self.wav_idx = np.array(wav_idx_)[idx_ascending]
            self.text_idx = np.array(text_idx_)[idx_ascending, :]
        else:
            self.batch_size = 1
            self.wav_idx = np.array(self.wav_idx)
            self.text_idx = np.array(self.text_idx)

    def get_wav_length(self, idx: int) -> float:
        raise NotImplementedError()

    def shuffle(self, seed: int):
        rng = np.random.default_rng(seed)   # deterministic random number generator
        bs = self.batch_size
        len_ = len(self.wav_idx) // bs
        idx_random = np.arange(len_)
        rng.shuffle(idx_random)
        self.wav_idx[:len_ * bs] = self.wav_idx[:len_ * bs].reshape((len_, bs))[idx_random, :].reshape(-1)
        self.text_idx[:len_ * bs, :] = self.text_idx[:len_ * bs, :].reshape((len_, bs, -1))[idx_random, :, :].reshape(len_ * bs, -1)
    
    def get_text(self, idx: int) -> torch.LongTensor:
        # text shape: [text_len]
        text = self.text_idx[idx]
        text = self.textprocessor(text)
        return torch.LongTensor(text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.wav_idx)


class Dataset(_Dataset):
    def load_wav(self, idx: int, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        extension = f".{self.hp.extension}" if self.hp.extension else ""
        wav_dir = os.path.join(self.wav_dir, f"{self.wav_idx[idx]}{extension}")
        wav, sr = librosa.core.load(wav_dir, sr=sr)
        if getattr(self.hp, "trim", False):
            wav, _ = librosa.effects.trim(wav, top_db=self.hp.trim_db, frame_length=800, hop_length=200)
        return wav, sr
    
    def get_wav_length(self, idx: int) -> float:
        extension = f".{self.hp.extension}" if self.hp.extension else ""
        wav_dir = os.path.join(self.wav_dir, f"{self.wav_idx[idx]}{extension}")
        try:
            with wave.open(wav_dir) as f:
                return f.getnframes() / f.getframerate()
        except:
            wav, sr = librosa.core.load(wav_dir, sr=None)
            return len(wav) / sr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}
        if "filename" in self.keys:
            data["filename"] = self.wav_idx[idx]
        
        if "text" in self.keys:
            text = self.get_text(idx)
            data["text"] = text     # shape: [num_mels, mel_len]
        if "text_len" in self.keys:
            data["text_len"] = text.size(0)

        wav, _ = self.load_wav(idx, sr=self.hp.sampling_rate)
        if self.normalize_method == NormalizeMethods.MAX:
            wav = 0.99*wav / np.abs(wav).max()
        elif self.normalize_method == NormalizeMethods.RANDOMGAIN:
            high = min(self.random_gain_high, 0.99 / (np.abs(wav).max() + 1e-12))
            low = min(self.random_gain_low, high)
            wav = np.random.uniform(low, high) * wav
        wav = torch.from_numpy(wav).to(torch.float32)

        # pad 0 & make wav_len = multiple of hop_size
        if self.segment_size is None:
            wav_len = wav.size(0)
            hop_size = getattr(self.hp, "hop_size", 1)
            discard_len = wav_len - wav_len // hop_size * hop_size
            #wav = F.pad(wav, (2400, 2400 - discard_len))
            if discard_len > 0:
                wav = wav[:-discard_len]
            assert wav.size(0) % hop_size == 0
        else:
            if wav.size(0) >= self.segment_size:
                #start_idx = torch.randint(0, wav.size(0) - self.segment_size + 1, (1,))
                start_idx = random.randint(0, wav.size(0) - self.segment_size)  # 0 <= start_idx <= wav.size(-1) - segment_size
                wav = wav[start_idx:start_idx+self.segment_size]
            else:
                wav = F.pad(wav, (0, self.segment_size - wav.size(0)), value=0)

        if "wav" in self.keys:
            data["wav"] = wav       # shape: [wav_len]
        if "wav_len" in self.keys:
            data["wav_len"] = wav.size(0)
        
        if "mel" in self.keys or "mel_loss" in self.keys or "spec" in self.keys:
            spec = stft(wav.unsqueeze(0), self.hp.n_fft, self.hp.hop_size, self.hp.win_size, magnitude=True)
        if "spec" in self.keys:
            data["spec"] = spec.squeeze(0)
        if "spec_len" in self.keys:
            data["spec_len"] = spec.size(-1)
        if "mel" in self.keys:
            mel = spec_to_mel(
                spec, self.hp.n_fft, self.hp.n_mel, self.hp.sampling_rate, self.hp.mel_fmin, self.hp.mel_fmax, self.hp.clip_val
            ).squeeze(0)
            assert mel.size(1) * self.hp.hop_size == wav.size(0)

            if self.hp.mel_normalize:
                mel = (mel - self.hp.mel_mean) / self.hp.mel_std
            data["mel"] = mel       # shape: [num_mels, mel_len]
        if "mel_loss" in self.keys:
            mel = spec_to_mel(
                spec, self.hp.n_fft, self.hp.n_mel, self.hp.sampling_rate, self.hp.mel_fmin, self.hp.mel_fmax_loss, self.hp.clip_val
            ).squeeze(0)
            data["mel_loss"] = mel  # shape: [num_mels, mel_len]
        if "mel_len" in self.keys:
            data["mel_len"] = mel.size(-1)
        
        if "pitch" in self.keys:
            fmin, fmax = 75, 600
            padding = math.floor(self.hp.sampling_rate / fmin * 3 / 2 - self.hp.hop_size / 2) + 1
            _wav = np.pad(wav.numpy(), (padding, padding))
            snd = parselmouth.Sound(_wav, self.hp.sampling_rate)
            spec_len = wav.size(0) // self.hp.hop_size

            pitch = snd.to_pitch(
                time_step=self.hp.hop_size/self.hp.sampling_rate,
                pitch_floor=fmin,
                pitch_ceiling=fmax
            ).selected_array['frequency']
            
            voiced = np.sign(pitch, dtype=np.float32)

            start_f0 = pitch[pitch != 0][0]
            end_f0 = pitch[pitch != 0][-1]
            start_idx = np.where(pitch == start_f0)[0][0]
            end_idx = np.where(pitch == end_f0)[0][-1]
            pitch[:start_idx] = start_f0
            pitch[end_idx:] = end_f0

            # get non-zero frame index
            nonzero_idxs = np.where(pitch != 0.)[0]

            # perform linear interpolation
            interp_fn = interp1d(nonzero_idxs, pitch[nonzero_idxs])
            pitch = interp_fn(np.arange(0, pitch.shape[0]))
            
            if self.hp.log_pitch:
                pitch = np.log(pitch)
            if self.hp.pitch_normalize:
                pitch = (pitch - self.hp.pitch_mean) / self.hp.pitch_std

            pitch = pitch.astype(np.float32)    # pitch > 0
            pitch = torch.from_numpy(pitch).unsqueeze(0)
            voiced = torch.from_numpy(voiced).unsqueeze(0)
            assert pitch.size(1) == spec_len, f"filename: {self.wav_idx[idx]}, padding: {padding}, pitch: {pitch.shape}, mel: {spec_len}, wav: {wav.shape}"
            
            data["pitch"] = pitch       # shape: [1, mel_len]
            data["voiced"] = voiced     # shape: [1, mel_len]
        
        return data


class DatasetPreprocessed(_Dataset):
    def __init__(self, hp, keys, textprocessor=None, mode="train", batch_size=1, verbose=True):
        super().__init__(hp, keys, textprocessor, mode, batch_size, verbose)
        if "mel" in self.keys or "spec" in self.keys:
            self.tail = getattr(hp, "tail", None)
            if self.tail is None:
                self.tail = "none" if hp.mel_fmax is None else f"{int(hp.mel_fmax/1000)}k"
        if "opus" in self.keys:
            self.opus_tails = hp.opus_tails
    
    def get_wav_length(self, idx: int) -> float:
        wav_dir = os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy")
        return os.path.getsize(wav_dir) / (4 * self.hp.sampling_rate)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}
        mel_start_idx = None
        if self.segment_size is not None:
            wav_ = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy"))
            wav = torch.from_numpy(wav_)
            if wav.size(0) >= self.segment_size:
                if any([x in self.keys for x in ["mel", "mel_loss", "pitch", "spec"]]):
                    mel_size = wav.size(0) // self.hp.hop_size
                    mel_seg_size = self.segment_size // self.hp.hop_size
                    mel_start_idx = random.randint(0, mel_size - mel_seg_size)
                    wav_start_idx = mel_start_idx * self.hp.hop_size
                else:
                    wav_start_idx = random.randint(0, wav.size(0) - self.segment_size)
                wav = wav[wav_start_idx:wav_start_idx+self.segment_size]
            else:
                wav_pad_len = self.segment_size - wav.size(0)
                wav = F.pad(wav, (0, wav_pad_len), value=0)
                if any([x in self.keys for x in ["mel", "mel_loss", "pitch", "spec"]]):
                    mel_pad_len = wav_pad_len // self.hp.hop_size
                    mel_pad_value = math.log(self.hp.clip_val)
        if "filename" in self.keys:
            data["filename"] = self.wav_idx[idx]
        if "text" in self.keys:
            text = self.get_text(idx)
            data["text"] = text     # shape: [num_mels, mel_len]
        if "text_len" in self.keys:
            data["text_len"] = text.size(0)
        if "wav" in self.keys:
            if self.segment_size is None:
                wav = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_wav.npy"))
                wav = torch.from_numpy(wav)
            data["wav"] = wav       # shape: [wav_len]
        if "wav_len" in self.keys:
            data["wav_len"] = wav.size(0)
        if "spec" in self.keys:
            spec = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_spec.npy"))
            spec = torch.from_numpy(spec)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    spec = F.pad(spec, (0, mel_pad_len), value=mel_pad_value)
                else:
                    spec = spec[..., mel_start_idx:mel_start_idx+mel_seg_size]
            data["spec"] = spec       # shape: [num_mels, mel_len]
        if "spec_len" in self.keys:
            data["spec_len"] = spec.size(-1)
        if "mel" in self.keys:
            mel = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_mel_{self.tail}.npy"))
            mel = torch.from_numpy(mel)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    mel = F.pad(mel, (0, mel_pad_len), value=mel_pad_value)
                else:
                    mel = mel[..., mel_start_idx:mel_start_idx+mel_seg_size]
            if self.hp.mel_normalize:
                mel = (mel - self.hp.mel_mean) / self.hp.mel_std
            data["mel"] = mel       # shape: [num_mels, mel_len]
        if "mel_loss" in self.keys:
            tail = "none" if self.hp.mel_fmax_loss is None else f"{int(self.hp.mel_fmax_loss/1000)}k"
            mel = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_mel_{tail}.npy"))
            mel = torch.from_numpy(mel)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    mel = F.pad(mel, (0, mel_pad_len), value=mel_pad_value)
                else:
                    mel = mel[..., mel_start_idx:mel_start_idx+mel_seg_size]
            data["mel_loss"] = mel  # shape: [num_mels, mel_len]
        if "mel_len" in self.keys:
            data["mel_len"] = mel.size(-1)
        if "pitch" in self.keys:
            pitch = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_pitch.npy"))
            voiced = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_voiced.npy"))
            pitch, voiced = torch.from_numpy(pitch), torch.from_numpy(voiced)
            if self.segment_size is not None:
                if mel_start_idx is None:
                    pitch = F.pad(pitch, (0, mel_pad_len), value=pitch[-1])
                    voiced = F.pad(voiced, (0, mel_pad_len), value=0)
                else:
                    pitch = pitch[..., mel_start_idx:mel_start_idx+mel_seg_size]
                    voiced = voiced[..., mel_start_idx:mel_start_idx+mel_seg_size]
            if self.hp.log_pitch:
                pitch = torch.log(pitch)
            if self.hp.pitch_normalize:
                pitch = (pitch - self.hp.pitch_mean) / self.hp.pitch_std
            
            data["pitch"] = pitch       # shape: [1, mel_len]
            data["voiced"] = voiced     # shape: [1, mel_len]
        if "dur" in self.keys:
            dur = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_{self.hp.duration_tail}.npy"))
            dur = torch.from_numpy(dur)
            data["dur"] = dur
        if "opus" in self.keys:
            tail = random.sample(self.opus_tails, k=1)[0]
            try:
                opus = np.load(os.path.join(self.data_dir, f"{self.wav_idx[idx]}_opus_{tail}.npy"))
            except Exception as e:
                print("error at file", self.wav_idx[idx])
                print(e)
                exit()
            opus = torch.from_numpy(opus)
            if self.segment_size is not None:
                if opus.size(0) >= self.segment_size:
                    opus = opus[wav_start_idx:wav_start_idx+self.segment_size]
                    opus = opus.to(torch.float32) / 2**15
                else:
                    opus = opus.to(torch.float32) / 2**15
                    opus = F.pad(opus, (0, wav_pad_len), value=0)
            else:
                opus = opus.to(torch.float32) / 2**15
            data["opus"] = opus
        
        return data


def collate(list_of_dicts: List[Dict[str, Any]]) -> TensorDict:
    data = TensorDict()
    batch_size = len(list_of_dicts)
    keys = list_of_dicts[0].keys()

    for key in keys:
        if key == "filename":
            data["filename"] = [x["filename"] for x in list_of_dicts]
            continue
        elif key.endswith("_len"):
            data[key] = torch.LongTensor([x[key] for x in list_of_dicts])
            continue
        max_len = max([x[key].size(-1) for x in list_of_dicts])
        tensor = torch.zeros(batch_size, *[x for x in list_of_dicts[0][key].shape[:-1]], max_len, dtype=list_of_dicts[0][key].dtype)
        for i in range(batch_size):
            value = list_of_dicts[i][key]
            tensor[i, ..., :value.size(-1)] = value
        data[key] = tensor
    return data


class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train", batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        self.clean_dir = hp.clean_dir
        self.noisy_dir = hp.noisy_dir
        self.segment_size = None if mode == "infer" else getattr(hp, "segment_size", None)
        
        self.files = []
        self.length_warned = False
        with open(hp.filelists[mode], 'r') as filelist:
            for file in filelist.readlines():
                file = file.strip()
                if file != "":
                    self.files.append(file)

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)

    def _wave_backend(self, clean_file: str, noisy_file: str) -> Tuple[np.ndarray, np.ndarray]:
        wave_read = wave.open(clean_file)
        assert wave_read.getframerate() == self.hp.sampling_rate
        wav_len = wave_read.getnframes()
        if wav_len == 0:
            raise RuntimeError(f"Empty audio {file}.")
        if self.segment_size is not None:
            if wav_len < self.segment_size:
                segment = wave_read.readframes(wav_len)
                wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
                padding = self.segment_size - len(wav)
                wav = np.pad(
                    wav,
                    (padding//2, padding - padding//2),
                    "constant"
                )
            else:
                start_idx = random.randint(0, wav_len - self.segment_size)
                wave_read.setpos(start_idx)
                segment = wave_read.readframes(self.segment_size)
                wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
        else:
            segment = wave_read.readframes(wav_len)
            wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
        clean = wav.astype(np.float32) / 32768.0
        
        wave_read = wave.open(noisy_file)
        assert wave_read.getnframes() == wav_len
        if self.segment_size is not None:
            if wav_len < self.segment_size:
                wav = np.pad(
                    wav,
                    (padding//2, padding - padding//2),
                    "constant"
                )
            else:
                wave_read.setpos(start_idx)
                segment = wave_read.readframes(self.segment_size)
                wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
        else:
            segment = wave_read.readframes(wav_len)
            wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
        noisy = wav.astype(np.float32) / 32768.0
        return clean, noisy
    
    def _librosa_backend(self, clean_file: str, noisy_file: str) -> Tuple[np.ndarray, np.ndarray]:
        clean, sr = librosa.core.load(clean_file, sr=None)
        noisy, sr2 = librosa.core.load(noisy_file, sr=None)
        assert sr == sr2 == self.hp.sampling_rate, \
            f"clean.sr: {sr}, noisy.sr: {sr2}, hp.data.sampling_rate: {self.hp.sampling_rate}"
        assert (len(clean) > 0) and (len(clean) == len(noisy)), \
            f"clean: {len(clean)}, noisy: {len(noisy)}"
        if self.segment_size is not None:
            if clean.size >= self.segment_size:
                start_idx = random.randint(0, clean.size - self.segment_size)  # 0 <= start_idx <= wav.size(-1) - segment_size
                clean = clean[start_idx:start_idx+self.segment_size]
                noisy = noisy[start_idx:start_idx+self.segment_size]
            else:
                if not self.length_warned:
                    print(f"segment_size {self.segment_size} is longer than the data size {clean.size}")
                    self.length_warned = True
                padding = self.segment_size - clean.size
                clean = np.pad(clean, (padding // 2, padding - padding // 2))
                noisy = np.pad(noisy, (padding // 2, padding - padding // 2))
        return clean, noisy
    
    def __getitem__(self, idx):
        file = self.files[idx]
        id = file.split("_")[-1]
        clean_file = os.path.join(self.clean_dir, f"clean_fileid_{id}")
        noisy_file = os.path.join(self.noisy_dir, file)
        try:
            clean, noisy = self._wave_backend(clean_file, noisy_file)
        except Exception as e:
            print(e)
            clean, noisy = self._librosa_backend(clean_file, noisy_file)

        return {"clean": clean, "noisy": noisy}


def collate_aec_testset(list_of_dicts: List[Dict[str, np.ndarray|int]]) -> TensorDict:
    data = TensorDict()
    batch_size = len(list_of_dicts)
    keys = list_of_dicts[0].keys()

    for key in keys:
        if key == "length":
            data[key] = torch.LongTensor([x[key] for x in list_of_dicts])
            continue
        max_len = max([x[key].size for x in list_of_dicts])
        tensor = np.zeros((batch_size, max_len), dtype=np.float32)
        for i in range(batch_size):
            value = list_of_dicts[i][key]
            tensor[i, :value.size] = value
        data[key] = torch.from_numpy(tensor)
    return data


class AECTestSet(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None):
        # hp = {
        #   filelist: "..."
        #   sampling_rate: 16000
        # }
        super().__init__()
        self.sampling_rate: int = hp.sampling_rate
        if keys is None:
            self.keys = ["far", "mix", "length"]
        else:
            self.keys: List[str] = keys
        
        self.files: List[str] = []
        self.length_warned = False
        with open(hp.erle_filelist, 'r') as filelist:
            for file in filelist.readlines():
                file = file.strip()
                if file != "":
                    self.files.append(file)

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        data = {}
        _len = -1
        if "far" in self.keys:
            file_far = file.replace("mic.wav", "lpb.wav").replace("mic_c.wav", "lpb.wav")
            far, sr = librosa.core.load(file_far, sr=None)
            assert sr == self.sampling_rate
            _len = far.size
            data["far"] = far
        if "mix" in self.keys:
            mix, sr = librosa.core.load(file, sr=None)
            assert sr == self.sampling_rate, f"wav.sr: {sr}, hp.data.sampling_rate: {self.sampling_rate}"
            if _len >= 0:
                if mix.size > _len:
                    print(f"{file}  - mix: {mix.size}, far: {_len}")
                    mix = mix[:_len]
                elif _len - 480 <= mix.size < _len:
                    _len = mix.size
                    data["far"] = data["far"][:_len]
                elif mix.size < _len - 480:
                    raise RuntimeError(f"mix.size: {mix.size}, far.size: {_len}")
            else:
                _len = mix.size
            data["mix"] = mix
        if "speex_error" in self.keys:
            file_error = file.replace("mic.wav", "speex_error.wav").replace(
                "mic_c.wav", "speex_error.wav")
            error, sr = librosa.core.load(file_error, sr=None)
            assert sr == self.sampling_rate
            data["speex_error"] = error
            if _len >= 0:
                assert error.size == _len
        
        if "length" in self.keys:
            assert _len >= 0
            data["length"] = _len

        return data


def get_metadata(metadata_path: str, metadata: str) -> List[float]:
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    fileid_idx = header.index("fileid")
    data_idx = header.index(metadata)
    fileid_data_list: List[Tuple[int, float]] = []
    for line in lines[1:]:
        data = line.strip().split(",")
        fileid = int(data[fileid_idx])
        value = float(data[data_idx])
        fileid_data_list.append((fileid, value))
    
    # for sanity check
    fileid_data_sorted = sorted(fileid_data_list, key=lambda x: x[0])
    for idx in range(len(fileid_data_sorted)):
        fileid = fileid_data_sorted[idx][0]
        if fileid != idx:
            raise RuntimeError(f"fileid: {fileid} / idx: {idx}")
    data = [x for _, x in fileid_data_sorted]
    return data


class AECDatasetBJWoo(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train",
                 batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        if keys is None:
            self.keys = ["near", "far", "mix"]
        else:
            self.keys: List[str] = keys
        self.segment_size = getattr(hp, "segment_size", None)
        self.sampling_rate = hp.sampling_rate
        
        self.length_warned = False

        if mode == "train":
            self.files = list(range(hp.train_idx.start, hp.train_idx.end + 1))
        elif mode == "valid":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
        elif mode == "infer":
            self.files = list(hp.infer_idx)
            self.segment_size = 160000
            self.length_warned = True
        elif mode == "pesq":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
            self.segment_size = 160000
            self.length_warned = True
        self.files: List[int]

        self.nearend_scale: List[float] = []
        if "nearend_scale" in self.keys:
            self.nearend_scale = get_metadata(hp.metadata, "nearend_scale")
        self.ser: List[float] = []
        if "ser" in self.keys:
            self.ser = get_metadata(hp.metadata, "ser")

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files) * self.mult
    
    def __getitem__(self, idx):
        file_idx = idx // self.mult
        seg_idx = idx % self.mult
        _id = self.files[file_idx]
        data = {}

        if "near" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.near_dir,
                    f"nearend_speech_fileid_{_id}.wav"),
                sr=None
            )
            data["near"] = x
            assert sr == self.sampling_rate

        if "far" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.far_dir,
                    f"farend_speech_fileid_{_id}.wav"),
                sr=None
            )
            data["far"] = x
            assert sr == self.sampling_rate

        if "mix" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.mix_dir,
                    f"nearend_mic_fileid_{_id}.wav"),
                sr=None
            )
            data["mix"] = x
            assert sr == self.sampling_rate

        if "echo" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.echo_dir,
                    f"echo_fileid_{_id}.wav"),
                sr=None
            )
            data["echo"] = x
            assert sr == self.sampling_rate

        if "speex_error" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.speex_error_dir,
                    f"nearend_mic_fileid_{_id}.wav"),
                sr=None
            )
            data["speex_error"] = x
            assert sr == self.sampling_rate

        if self.segment_size is not None:
            start_idx = 16000 * seg_idx
            end_idx = start_idx + self.segment_size
            if x.size >= end_idx:
                for key, value in data.items():
                    data[key] = value[start_idx:end_idx]
            else:
                if not self.length_warned:
                    print(f"Warning - end_idx {end_idx}"
                          f" is longer than the data size {x.size}")
                    self.length_warned = True
                padding = end_idx - x.size
                for key, value in data.items():
                    value = np.pad(value[start_idx:end_idx], (0, padding))
                    data[key] = value

        if "nearend_scale" in self.keys:
            data["nearend_scale"] = np.array([self.nearend_scale[_id]], dtype=np.float32)

        return data


class AECDataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train",
                 batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        if keys is None:
            self.keys = ["near", "far", "mix"]
        else:
            self.keys: List[str] = keys
        self.segment_size = getattr(hp, "segment_size", None)
        self.sampling_rate = hp.sampling_rate
        
        self.length_warned = False

        if mode == "train":
            self.files = list(range(hp.train_idx.start, hp.train_idx.end + 1))
        elif mode == "valid":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
        elif mode == "infer":
            self.files = list(hp.infer_idx)
            self.segment_size = 160000
            self.length_warned = True
        elif mode == "pesq":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
            self.segment_size = 160000
            self.length_warned = True
        self.files: List[int]

        self.nearend_scale: List[float] = []
        if "nearend_scale" in self.keys:
            self.nearend_scale = get_metadata(hp.metadata, "nearend_scale")
        self.ser: List[float] = []
        if "ser" in self.keys:
            self.ser = get_metadata(hp.metadata, "ser")
        self.is_nearend_noisy: List[float] = []
        if "is_nearend_noisy" in self.keys:
            self.is_nearend_noisy = get_metadata(hp.metadata, "is_nearend_noisy")

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _id = self.files[idx]
        data = {}
        
        if "label" in self.keys:
            y = np.load(
                os.path.join(self.hp.label_dir,
                f"nearend_mic_fileid_{_id}.npy")
            )
            data["label"] = y

        if "near" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.near_dir,
                    f"nearend_speech_fileid_{_id}.wav"),
                sr=None
            )
            data["near"] = x
            assert sr == self.sampling_rate

        if "far" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.far_dir,
                    f"farend_speech_fileid_{_id}.wav"),
                sr=None
            )
            data["far"] = x
            assert sr == self.sampling_rate

        if "mix" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.mix_dir,
                    f"nearend_mic_fileid_{_id}.wav"),
                sr=None
            )
            data["mix"] = x
            assert sr == self.sampling_rate

        if "echo" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.echo_dir,
                    f"echo_fileid_{_id}.wav"),
                sr=None
            )
            data["echo"] = x
            assert sr == self.sampling_rate

        if "speex_error" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.speex_error_dir,
                    f"nearend_mic_fileid_{_id}.wav"),
                sr=None
            )
            data["speex_error"] = x
            assert sr == self.sampling_rate

        if self.segment_size is not None:
            if x.size >= self.segment_size:
                # 0 <= start_idx <= wav.size(-1) - segment_size
                start_idx_label = random.randint(0, (x.size - self.segment_size) // 160)
                start_idx_others = start_idx_label * 160
                for key, value in data.items():
                    if key == "label":
                        start_idx = start_idx_label
                        segment_size = self.segment_size // 160 + 1
                    else:
                        start_idx = start_idx_others
                        segment_size = self.segment_size
                    data[key] = value[start_idx : start_idx+segment_size]
            else:
                if not self.length_warned:
                    print(f"Warning - segment_size {self.segment_size}"
                          f" is longer than the data size {x.size}")
                    self.length_warned = True
                padding = self.segment_size - x.size
                for key, value in data.items():
                    if key == "label":
                        continue
                    data[key] = np.pad(
                        value,
                        (padding // 2, padding - padding // 2)
                    )

        if "nearend_scale" in self.keys:
            data["nearend_scale"] = np.array([self.nearend_scale[_id]], dtype=np.float32)
        if "ser" in self.keys:
            data["ser"] = np.array([self.ser[_id]], dtype=np.float32)
        if "is_nearend_noisy" in self.keys:
            data["is_nearend_noisy"] = np.array([self.is_nearend_noisy[_id]], dtype=bool)

        return data


class AECFBDataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train",
                 batch_size=1, verbose=False):
        super().__init__()
        self.hp = hp
        self.base_dir: str = hp.fullband_dir
        if keys is None:
            self.keys = ["near", "far", "mix"]
        else:
            self.keys: List[str] = keys
        self.segment_size = getattr(hp, "segment_size", None)
        self.sampling_rate = hp.sampling_rate
        
        self.length_warned = False

        if mode == "train":
            self.files = list(range(hp.train_idx.start, hp.train_idx.end + 1))
        elif mode == "valid":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
        elif mode == "infer":
            self.files = list(hp.infer_idx)
            self.segment_size = 480000
            self.length_warned = True
        elif mode == "pesq":
            self.files = list(range(hp.valid_idx.start, hp.valid_idx.end + 1))
            self.segment_size = 480000
            self.length_warned = True
        self.files: List[int]

    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _id = self.files[idx]
        data = {}

        if "near" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.base_dir, f"f{_id:0>5d}_target.wav"),
                sr=None
            )
            data["near"] = x
            assert sr == self.sampling_rate

        if "near_noisy" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.base_dir, f"f{_id:0>5d}_nearend.wav"),
                sr=None
            )
            data["near_noisy"] = x
            assert sr == self.sampling_rate

        if "far" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.base_dir, f"f{_id:0>5d}_farend.wav"),
                sr=None
            )
            data["far"] = x
            assert sr == self.sampling_rate

        if "mix" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.base_dir, f"f{_id:0>5d}_mic.wav"),
                sr=None
            )
            data["mix"] = x
            assert sr == self.sampling_rate

        if "echo" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.base_dir, f"f{_id:0>5d}_echo.wav"),
                sr=None
            )
            data["echo"] = x
            assert sr == self.sampling_rate

        if "speex_error" in self.keys:
            x, sr = librosa.core.load(
                os.path.join(self.hp.speex_error_dir,
                    f"nearend_mic_fileid_{_id}.wav"),
                sr=None
            )
            data["speex_error"] = x
            assert sr == self.sampling_rate

        if self.segment_size is not None:
            if x.size >= self.segment_size:
                # 0 <= start_idx <= wav.size(-1) - segment_size
                start_idx = random.randint(0, x.size - self.segment_size)
                for key, value in data.items():
                    data[key] = value[start_idx:start_idx+self.segment_size]
            else:
                if not self.length_warned:
                    print(f"Warning - segment_size {self.segment_size}"
                          f" is longer than the data size {x.size}")
                    self.length_warned = True
                padding = self.segment_size - x.size
                for key, value in data.items():
                    data[key] = np.pad(
                        value,
                        (padding // 2, padding - padding // 2)
                    )

        return data
