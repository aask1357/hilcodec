try:
    from utils.visqol_python import measure_visqol
except:
    from utils.visqol_cli import measure_visqol

if __name__=="__main__":
    # ref = librosa.core.load("/home/shahn/Datasets/AEC-Challenge/near/nearend_speech_fileid_0.wav", sr=16000)[0]
    # deg = librosa.core.load("/home/shahn/Datasets/AEC-Challenge/mix/nearend_mic_fileid_0.wav", sr=16000)[0]
    ref = librosa.core.load("/home/shahn/Documents/visqol/.cache/ref_0.wav", sr=16000)[0]
    deg = librosa.core.load("/home/shahn/Documents/visqol/.cache/deg_0.wav", sr=16000)[0]
    print(measure_visqol(ref, deg, 0, "speech"))
