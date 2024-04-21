from subprocess import check_output, DEVNULL
import re
import os
import soundfile as sf


VISQOL_BASE_DIR = "/home/shahn/Documents/visqol"
NUM_GPU_TOTAL = 8


def measure_visqol(ref, deg, idx: int = 0, mode: str = "speech") -> float:
    iii = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    cache_dir = os.path.join(VISQOL_BASE_DIR, ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if mode == "speech":
        sr = 16000
        postfix = " --use_speech_mode"
    elif mode == "audio":
        sr = 48000
        postfix = ""
    else:
        raise ValueError(f"Unrecognized visqol mode: {mode}")
    
    ref_path = os.path.join(cache_dir, f"ref_{idx * NUM_GPU_TOTAL + iii}.wav")
    deg_path = os.path.join(cache_dir, f"deg_{idx * NUM_GPU_TOTAL + iii}.wav")
    sf.write(ref_path, ref, sr)
    sf.write(deg_path, deg, sr)
    out = check_output(
        f"cd /home/shahn/Documents/visqol && "
        f"./bazel-bin/visqol --reference_file {ref_path} -degraded_file {deg_path}"
        f"{postfix}",
        shell=True,
        stderr=DEVNULL
    ).decode('utf-8')
    m = re.search("(?<=MOS-LQO:		)\d(\.\d+)?", out)
    if m:
        return float(m.group())
    else:
        print(out)
        raise RuntimeError("")

def test():
    print(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

if __name__=="__main__":
    ref = "/home/shahn/Datasets/AEC-Challenge/near/nearend_speech_fileid_0.wav"
    deg = "/home/shahn/Datasets/AEC-Challenge/mix/nearend_mic_fileid_0.wav"
    # ref = "/home/shahn/Documents/visqol/.cache/ref_0.wav"
    # deg = "/home/shahn/Documents/visqol/.cache/deg_0.wav"
    ref, _ = sf.read(ref)
    deg, _ = sf.read(deg)
    print(measure_visqol(ref, deg, 1, "speech"))
    test()
