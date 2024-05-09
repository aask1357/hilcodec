# Description
Official code for the paper "HILCodec: High Fidelity and Lightweight Neural Audio Codec".  
\[[paper]()\] \[[samples](https://aask1357.github.io/hilcodec/)\]

# Environment
We tested under CUDA=11.7, torch=1.13 and CUDA=10.2, torch=1.12.  
It may work in other environments, but not guaranteed.

# Install using anaconda
## Intall for training
First, install [PyTorch](https://pytorch.org/get-started/locally/) along with torchaudio.  
Then, install other requirements as below.
<pre><code>conda install librosa -c conda-forge
conda install jupyter notebook matplotlib scipy tensorboard tqdm pyyaml
pip install pesq pystoi</code></pre>
Finally, install [ONNXRuntime for CPU ](https://onnxruntime.ai/docs/install/).  
Optionally, install [ViSQOL](https://github.com/google/visqol).  
## Install for test
For test, you only need to install ONNXRuntime, librosa, and soundfile.  

# Datasets
Download VCTK, DNS-Challenge4 and Jamendo dataset for training.
For validation, we used `p225`, `p226`, `p227`, and `p228` from VCTK for clean speech. Real noisy speech recordings from DNS-Challenge4 are used for noisy speech. `Jamendo/99` are used for music.  
Downsample all audio files into 24khz before training (see `scripts/Resampling.ipynb`).  

# Training
Use `configs/...yaml` file to change configurations.  
Modify `directories_to_include`, `directories_to_exclude`, `wav_dir`.  
Also, modify `filelists/infer_24khz.txt` or `filelists/infer_speech.txt` file, which cotain audio files used for inference in tensorboard.  
Either use `train.py` or `train_torchrun.py` for training. Examples are:  
<pre><code>CUDA_VISIBLE_DEVICES=0,1 python train.py -c configs/hilcodec_music.yaml -n first_exp -p train.batch_size=16 train.seed=1234 -f</code></pre>
<pre><code>CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_torchrun.py -c configs/hilcodec_music.yaml -n first_exp -p train.batch_size=16 train.seed=1234 -f</code></pre>  
Arguments:  
-n: (Required) Directory name to save checkpoints, the configuration file, and tensorboard logs.  
-c: (Optional) Configuration file path. If not given, use a configuration file in the directory.  
-p: (Optional) Parameters after this will update configurations.  
-f: (Optional) If the directory already exists, an exception will be raised to avoid overwriting config file. However, enabling this option will force overwriting config file.

# Inference
Pre-trained model parameters are provided in the `onnx` directory. Two versions are available: 
- hilcodec_music  
- hilcodec_speech  

Modify the variable `PATH` in `test_onnx.py` as you want, and run the following code:
<pre><code>python test_onnx.py -n hil_speech --enc --dec</code></pre>  
The output will be saved at `onnx/hil_speech_output.wav`.  
Use `python test_onnx.py --help` for information about each argument.  
Note that for AudioDec, you must set `-H 300`.  

You can convert your own trained HILCodec to ONNXRuntime using `scripts/HILCodec Onnx.ipynb`.  
You can also convert [Encodec](https://github.com/facebookresearch/encodec) and [AudioDec](https://github.com/facebookresearch/AudioDec) to ONNXRuntime for comparison.  
Download checkpoints from official repositories and use `scripts/Encodec Onnx.ipynb` or `scripts/AudioDec Onnx.ipynb`.
script. It saves logs in `logs/first` directory.   

# Evaluating PESQ, STOI and ViSQOL
Our training code includes objective metrics calculation. You can set `pesq` in a config file appropriately.  
Note that on our server it occasionally crashes (especially when calculating ViSQOL), so the default config is to turn off calculation.  
To calculate metrics after training, you can use `scripts/pesq.ipynb`.  
