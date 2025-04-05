import os
import torchaudio

DATASET_PATH = './mixed_vox2'
SAMPLE_RATE = 8000

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            waveform, sr = torchaudio.load(file_path)
            if sr != SAMPLE_RATE:
                print(f"Resampling {file_path} from {sr} Hz to {SAMPLE_RATE} Hz")
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
                torchaudio.save(file_path, waveform, SAMPLE_RATE)
