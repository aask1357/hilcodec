import os
import librosa
import numpy as np

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2


config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 48000
config.options.use_speech_scoring = False
svr_model_path = "libsvm_nu_svr_model.txt"
config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
AudioAPI = visqol_lib_py.VisqolApi()
AudioAPI.Create(config)

config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 16000
config.options.use_speech_scoring = True
svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
config.options.use_lattice_model = True
SpeechAPI = visqol_lib_py.VisqolApi()
with open(os.devnull, 'w') as f:    # prevent stderr from being printed
    saved_stderr = os.dup(2)
    os.dup2(f.fileno(), 2)
    SpeechAPI.Create(config)
    os.dup2(saved_stderr, 2)


def measure_visqol(ref: np.ndarray, deg: np.ndarray, idx: int, mode: str) -> float:
    if mode == "audio":
        api = AudioAPI
    elif mode == "speech":
        api = SpeechAPI
    else:
        raise ValueError(f"Unrecognized visqol mode: {mode}")

    similarity_result = api.Measure(ref.astype(np.float64), deg.astype(np.float64))
    return similarity_result.moslqo


if __name__=="__main__":
    # ref = librosa.core.load("/home/shahn/Datasets/AEC-Challenge/near/nearend_speech_fileid_0.wav", sr=16000)[0]
    # deg = librosa.core.load("/home/shahn/Datasets/AEC-Challenge/mix/nearend_mic_fileid_0.wav", sr=16000)[0]
    ref = librosa.core.load("/home/shahn/Documents/visqol/.cache/ref_0.wav", sr=16000)[0]
    deg = librosa.core.load("/home/shahn/Documents/visqol/.cache/deg_0.wav", sr=16000)[0]
    print(measure_visqol(ref, deg, 0, "speech"))
