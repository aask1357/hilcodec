import argparse
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime
import librosa
import soundfile as sf


PATH = "/home/shahn/Datasets/LibriSpeech/test-clean/2094/142345/2094-142345-0010.flac"


class Timer:
    def __init__(self, sr: int):
        self.sr = sr
        self.enc_time = 0.0
        self.dec_time = 0.0
        self.start_time = time.perf_counter()
        self.wav_len = 0
    
    def tic(self):
        self.start_time = time.perf_counter()
    
    def encoder_time(self):
        et = time.perf_counter()
        self.enc_time += et - self.start_time
        self.start_time = et
    
    def decoder_time(self):
        et = time.perf_counter()
        self.dec_time += et - self.start_time
        self.start_time = et
    
    def print(self):
        wav_time = self.wav_len / self.sr
        print(f"\rwav length: {wav_time:.1f} s")
        if self.enc_time > 0:
            print(f"encoder: {self.enc_time:.1f} s / rtf: {wav_time/self.enc_time:.4f} (↑)")
        if self.dec_time > 0:
            print(f"decoder: {self.dec_time:.1f} s / rtf: {wav_time/self.dec_time:.4f} (↑)")


def encoder(name: str, hop_size: int, num_quantizers: int, timer: Timer, sr: int,
            so: onnxruntime.SessionOptions):
    wav, _ = librosa.load(PATH, sr=sr)
    wav = np.concatenate((wav, wav))
    length = len(wav) // hop_size * hop_size
    wav = wav[np.newaxis, np.newaxis, :length]
    timer.wav_len = length

    ##### Check model #####
    onnx_model = onnx.load(f"onnx/{name}_enc.onnx")
    onnx.checker.check_model(onnx_model)
    for i in range(num_quantizers):
        onnx_model = onnx.load(f"onnx/{name}_vq{i}.onnx")
        onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    ##### Load model & initialize input #####
    enc = onnxruntime.InferenceSession(f"onnx/{name}_enc.onnx", sess_options=so)
    vq = dict()
    for i in range(num_quantizers):
        vq[i] = onnxruntime.InferenceSession(f"onnx/{name}_vq{i}.onnx", sess_options=so)
    enc_input = dict(np.load(f"onnx/{name}_cache_enc.npz"))

    indices_list = [[] for _ in range(num_quantizers)]
    timer.tic()
    for i in tqdm(range(0, length, hop_size), desc="Enc+Q", ncols=80):
    # for i in range(0, length, hop_size):
        # Encoder
        enc_input["wav_in"] = wav[:, :, i:i+hop_size]
        out = enc.run(None, enc_input)
        x = out[0]
        for i in range(len(out[1:])):
            enc_input[f"e_in{i}"] = out[i+1]
        
        # VQ
        residual = x
        quantized, index = vq[0].run(None, {"x": residual})
        # index: [B, F] where F is the number of frames
        quantized_out = quantized
        indices_list[0].append(index)
        for j in range(1, num_quantizers):
            residual = residual - quantized
            quantized, index = vq[j].run(None, {"x": residual})
            quantized_out += quantized
            indices_list[j].append(index)
    timer.encoder_time()
    
    # indices_list[i]: T//F x [B, F]
    for i in range(num_quantizers):
        indices_list[i] = np.concatenate(indices_list[i], 1)    # [B, T]
    indices = np.stack(indices_list).astype(np.int16)  # [n, B, T]
    np.save(f"onnx/{name}_quantized.npy", indices)


def decoder(name: str, num_frames: int, num_quantizers: int, timer: Timer, sr: int,
            so: onnxruntime.SessionOptions):
    indices = np.load(f"onnx/{name}_quantized.npy").astype(np.int64)

    ##### Check model #####
    onnx_model = onnx.load(f"onnx/{name}_dec.onnx")
    onnx.checker.check_model(onnx_model)
    for i in range(num_quantizers):
        onnx_model = onnx.load(f"onnx/{name}_deq{i}.onnx")
        onnx.checker.check_model(onnx_model)

    ##### Load model & initialize input #####
    dec = onnxruntime.InferenceSession(f"onnx/{name}_dec.onnx", sess_options=so)
    deq = dict()
    for i in range(num_quantizers):
        deq[i] = onnxruntime.InferenceSession(f"onnx/{name}_deq{i}.onnx", sess_options=so)
    dec_input = dict(np.load(f"onnx/{name}_cache_dec.npz"))

    wav_out_list = []
    timer.tic()
    for i in tqdm(range(0, indices.shape[2], num_frames), desc="Dec", ncols=80):
        # Dequantizer
        quantized = deq[0].run(None, {"idx": indices[0, :, i:i+num_frames]})[0]
        for j in range(1, num_quantizers):
            quantized += deq[j].run(None, {"idx": indices[j, :, i:i+num_frames]})[0]
        
        # Decoder
        dec_input["q"] = quantized
        out = dec.run(None, dec_input)
        wwv_out = out[0]
        for j in range(len(out[1:])):
            dec_input[f"d_in{j}"] = out[j+1]
        wav_out_list.append(np.squeeze(wwv_out))
    timer.decoder_time()
    wav_out = np.concatenate(wav_out_list)
    timer.wav_len = len(wav_out)
    sf.write(f"onnx/{name}_output.wav", wav_out, sr)


def main(
    name: str = "small",
    num_quantizers: int = 12,
    num_threads: int = 1,
    num_frames: int = 1,
    run_encoder: bool = True,
    run_decoder: bool = True,
    sr: int = 24_000,
    hop_size: int = 320
):
    ##### parameters #####
    hop_size = hop_size * num_frames
    num_quantizers = num_quantizers
    sess_options = onnxruntime.SessionOptions()
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads

    timer = Timer(sr=sr)
    if run_encoder:
        encoder(name, hop_size, num_quantizers, timer, sr, sess_options)
    if run_decoder:
        decoder(name, num_frames, num_quantizers, timer, sr, sess_options)
    timer.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="hil_speech",
                        help="Model name. Default: hil_speech")
    parser.add_argument("-q", "--num_quantizers", type=int, default=8,
                        help="Number of quantizers to use. Default: 8")
    parser.add_argument("-t", "--num_threads", type=int, default=1,
                        help="Number of threads to use. Default: 1")
    parser.add_argument("-f", "--num_frames", type=int, default=1,
                        help="Number of frames to process at once. Default: 1")
    parser.add_argument("-H", "--hop_size", type=int, default=320,
                        help="Hop size. Default: 320")
    parser.add_argument("--enc", action="store_true",
                        help="Run encoder")
    parser.add_argument("--dec", action="store_true",
                        help="Run decoder")
    parser.add_argument("--sr", type=int, default=24_000,
                        help="Sampling rate. Default: 24000")
    args = parser.parse_args()
    main(args.name, args.num_quantizers, args.num_threads, args.num_frames,
         args.enc, args.dec, args.sr, args.hop_size)
