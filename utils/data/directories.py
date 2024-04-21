import os
import math
import random
import typing as tp
from pathlib import Path
import wave

import numpy as np
from torch.utils.data import Dataset as TorchDataset
import librosa


class RandomGain:
    def __init__(self, low_db: float, high_db: float):
        self.low_db = low_db
        self.high_db = high_db
    
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(self.low_db, self.high_db)
        gain = 10**(gain_db / 20)   # dB -> amplitude# dB -> amplitude
        return wav * gain


AUDIO_EXT = [".wav", ".WAV", ".flac", ".FLAC", ".mp3"]  # extensions to detect as audio files


def is_audiofile(file: str) -> bool:
    return any(file.endswith(ext) for ext in AUDIO_EXT)


def is_directory_to_exclude(cur_dir: Path, blacklist: tp.List[str]) -> bool:
    parent_dir = cur_dir.parents
    for bl in blacklist:
        bl = Path(bl)
        if bl in parent_dir:
            return True
        if bl == cur_dir:
            return True
    return False


def is_file_to_exclude(file: Path, blacklist: tp.List[Path]) -> bool:
    return file in [Path(bl) for bl in blacklist]


class Directories:
    def __init__(
        self,
        directories_to_include: tp.List[str],
        directories_to_exclude: tp.List[str] = [],
        extension: str = "",
        mix: tp.Optional[tp.Dict[str, float]] = None,
        files_to_exclude: tp.List[Path] = [],
    ):
        self.extension = extension
        self.names_to_mix: tp.List[str] = []
        self.probabilities: tp.List[float] = []
        if mix is not None:
            for name, prob in mix.items():
                self.names_to_mix.append(name)
                self.probabilities.append(prob)
            self.names_to_mix.append("")
            self.probabilities.append(1.0 - sum(self.probabilities))
        
        self.dir_filelist: tp.Dict[str, tp.List[str]] = {}

        self.total_lengths = 0
        lengths = {}
        for directory in directories_to_include:
            file_list = []
            for root, _, files in os.walk(directory):
                root = Path(root)
                
                if is_directory_to_exclude(root, directories_to_exclude):
                    continue
                
                for file in files:
                    full_path = root / Path(file)
                    
                    if full_path in files_to_exclude:
                        continue
                    
                    file = str(full_path.relative_to(directory))
                    if extension == "":
                        if is_audiofile(file):
                            file_list.append(file)
                    elif file.endswith(extension):
                        file_list.append(file[:-len(extension)])

            if len(file_list) == 0:
                raise RuntimeError(
                    f"Directory {directory} has total_lengths: {len(file_list)},"
                    " but should be > 0")

            file_list.sort()
            self.dir_filelist[directory] = file_list
            self.total_lengths += len(file_list)
            lengths[directory] = len(file_list)

        sorted_lengths = sorted(lengths.items())
        self.lengths = {directory: length for directory, length in sorted_lengths}
    
    def choice(self) -> str:
        idx = random.randrange(self.total_lengths)
        cumsum = 0
        for directory, length in self.lengths.items():
            if idx < cumsum + length:
                file = self.dir_filelist[directory][idx - cumsum]
                full_path = os.path.join(directory, file + self.extension)
                return full_path
            cumsum += length
        raise RuntimeError(self.lengths, self.total_lengths, idx)


class DirectoriesDataset(TorchDataset):
    def __init__(self, hp, keys, textprocessor=None, mode="train", batch_size=1, verbose=True):
        super().__init__()
        assert hp.segment_size % 2 == 0, \
            f"segment_size must be divisible by 2, but got {hp.segment_size}"
        self.keys: tp.List[str] = keys
        self.segment_size: int = hp.segment_size
        self.sampling_rate: int = hp.sampling_rate
        self.length: int = hp.length
        self.fixed = False
        self.hash = np.arange(self.length)
        
        self.transforms_list = []
        for name, kwargs in hp.transforms.items():
            if name == "RandomGain":
                self.transforms_list.append(
                    RandomGain(kwargs.low_db, kwargs.high_db)
                )
            else:
                raise ValueError(f"Unknown transform: {name}")

        files_to_exclude = []
        for filelist in getattr(hp, "files_to_exclude", []):
            with open(filelist, "r") as f:
                files_to_exclude.extend(
                    [Path(file.rstrip()) for file in f.readlines()]
                )
        files_to_exclude = [Path(f) for f in files_to_exclude]
        
        self.loaders: tp.Dict[str, Directories] = {}
        self.directories: tp.List[Directories] = []
        self.probabilities: tp.List[float] = []
        cum_prob = 0.
        for name, kwargs in hp.classes.items():
            dirs = Directories(
                directories_to_include=kwargs.directories_to_include,
                directories_to_exclude=getattr(kwargs, "directories_to_exclude", []),
                extension=kwargs.extension,
                mix=getattr(kwargs, "mix", None),
                files_to_exclude=files_to_exclude,
            )
            self.loaders[name] = dirs
            self.directories.append(dirs)
            self.probabilities.append(kwargs.probability)
            cum_prob += kwargs.probability
        assert math.isclose(cum_prob, 1.0), f"cum_prob: {cum_prob}, but should  be 1.0"
    
    def shuffle(self, epoch: int) -> None:
        pass
    
    def transform(self, wav: np.ndarray) -> np.ndarray:
        for transform in self.transforms_list:
            wav = transform(wav)
        return wav
    
    def __len__(self) -> int:
        return self.length

    def _wave_backend(self, file: str, sr: int, segment_size: int) -> np.ndarray:
        wave_read = wave.open(file)
        assert wave_read.getframerate() == sr
        wav_len = wave_read.getnframes()
        if wav_len == 0:
            raise RuntimeError(f"Empty audio {file}.")
        if wav_len < segment_size:
            segment = wave_read.readframes(wav_len)
            wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
            padding = segment_size - len(wav)
            wav = np.pad(
                wav,
                (padding//2, padding - padding//2),
                "constant"
            )
        else:
            start_idx = random.randint(0, wav_len - segment_size)
            wave_read.setpos(start_idx)
            segment = wave_read.readframes(segment_size)
            wav = np.frombuffer(segment, dtype=np.int16, count=len(segment)//2, offset=0)
        return wav.astype(np.float32) / 32768.0
    
    def _librosa_backend(self, file: str, sr: int, segment_size: int) -> np.ndarray:
        wav, _ = librosa.core.load(file, sr=sr)
        wav_len = len(wav)
        if wav_len == 0:
            raise RuntimeError(f"Empty audio {file}.")
        if wav_len < segment_size:
            padding = segment_size - wav_len
            wav = np.pad(wav, (padding//2, padding - padding//2), "constant")
        elif wav_len > segment_size:
            # 0 <= start < wav_len - self.segment_size
            start = random.randrange(wav_len - segment_size)
            wav = wav[start:start + segment_size]
        return wav
    
    def load_wav(self, directories: Directories) -> tp.Tuple[np.ndarray, str]:
        error_count = 0
        while error_count < 10:
            error_count += 1
            filepath = directories.choice()
            if filepath.endswith(".wav") or filepath.endswith(".WAV"):
                try:
                    wav = self._wave_backend(filepath, self.sampling_rate, self.segment_size)
                    return wav, filepath
                except Exception as e:
                    pass
            try:
                wav = self._librosa_backend(filepath, self.sampling_rate, self.segment_size)
                return wav, filepath
            except Exception as e:
                pass
        raise RuntimeError(f"10 times failed to load wav from {directories}")
    
    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        data = {}
        dirs: Directories = np.random.choice(
            self.directories,         # type: ignore
            p=self.probabilities
        )
        
        wav, filepath = self.load_wav(dirs)
        wav = self.transform(wav)
        
        if dirs.names_to_mix:
            name_to_mix = np.random.choice(dirs.names_to_mix, p=dirs.probabilities)
            if name_to_mix:
                dirs = self.loaders[name_to_mix]
                wav_to_mix, filepath_to_mix = self.load_wav(dirs)
                wav_to_mix = self.transform(wav_to_mix)
                filepath = filepath +  " | " + filepath_to_mix
                wav = wav + wav_to_mix
        
        wav_max = np.max(np.abs(wav))
        if wav_max > 1.0:
            wav = wav / (wav_max + 1e-12)
        
        data["wav"] = wav
        
        if "filename" in self.keys:
            data["filename"] = filepath
        
        return data


if __name__=="__main__":
    import os
    os.environ["PYTHONPATH"] = "/home/shahn/Documents/trainer"
    from utils import HParams
    hp = HParams(
        segment_size = 16000,
        sampling_rate = 16000,
        length = 100,
        transforms = {
            "RandomGain": {
                "low_db": -10,
                "high_db": 6,
            },
        },
        classes = {
            "clean": {
                "directories_to_include": [
                    "/home/shahn/Datasets/DNS-Challenge3/datasets/wideband/clean_wideband",
                    "/home/shahn/Datasets/sitec/SiTEC_Dict01_reading_sentence/SD01-Dict01-Ver1.2_1-2/data/male/set001-030/mcn1lsg00s010"
                ],
                "directories_to_exclude": [
                    "/home/shahn/Datasets/AEC-Challenge2023/datasets/synthetic/echo_signal",
                ],
                "extension": ".wav",
                "probability": 0.5,
                "mix": {
                    "noise": 0.5,
                }
            },
            "noise": {
                "directories_to_include": [
                    "/home/shahn/Datasets/DNS-Challenge3/datasets/wideband/noise_wideband",
                ],
                "extension": ".wav",
                "probability": 0.5,
            }
        },
        
        files_to_exclude = [
            "filelists/DNS/format_error.txt",
            "filelists/DNS/empty_audio.txt"
        ],
    )
    dataset = DirectoriesDataset(hp, ["wav", "filename"])
    for idx in range(len(dataset)):
        if idx >= 0:
            break
        print(dataset[idx]["filename"])