import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from collections import defaultdict
from models.ecapa_tdnn import ECAPA_TDNN_SMALL


# ------------------ Config ------------------ #
MODEL_PATH = './wav2vec2_xlsr_SV_fixed.th'  # Your checkpoint
DATASET_PATH = './vox1_test_wav'            # Vox1-O test set
VERI_PATH = './veri_test2.txt'              # Verification pairs

# ------------------ Device setup ------------------ #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Load model ------------------ #
print("Loading ECAPA-TDNN model from checkpoint...")

from models.ecapa_tdnn import ECAPA_TDNN_SMALL

# Initialize your model
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None)

# Load checkpoint weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model = model.to(device).eval()

# ------------------ Utility functions ------------------ #

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)
    return waveform.to(device)

def extract_embedding(file_path):
    waveform = load_audio(file_path)
    with torch.no_grad():
        embedding = model(waveform)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy()

# ------------------ Build speaker embeddings ------------------ #

print("Extracting speaker embeddings...")
speaker_embeddings = defaultdict(list)

for speaker_id in tqdm(os.listdir(DATASET_PATH)):
    speaker_folder = os.path.join(DATASET_PATH, speaker_id)

    if not os.path.isdir(speaker_folder):
        continue  # Skip non-directories

    for utterance_folder in os.listdir(speaker_folder):
        utterance_folder_path = os.path.join(speaker_folder, utterance_folder)

        if not os.path.isdir(utterance_folder_path):
            continue  # Skip non-directories

        for file in os.listdir(utterance_folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(utterance_folder_path, file)
                emb = extract_embedding(file_path)
                speaker_embeddings[speaker_id].append(emb)


# Average embeddings per speaker for identification
speaker_avg_embeddings = {spk: np.mean(embeds, axis=0) for spk, embeds in speaker_embeddings.items()}

# ------------------ Verification evaluation ------------------ #

print("Running verification evaluation...")
scores = []
labels = []

with open(VERI_PATH, 'r') as f:
    for line in tqdm(f):
        label, path1, path2 = line.strip().split()
        full_path1 = os.path.join(DATASET_PATH, path1)
        full_path2 = os.path.join(DATASET_PATH, path2)
        emb1 = extract_embedding(full_path1)
        emb2 = extract_embedding(full_path2)
        score = np.dot(emb1, emb2)  # Cosine similarity
        scores.append(score)
        labels.append(int(label))

# Compute EER and TAR@1%FAR
fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] * 100
tar_at_1_far = tpr[np.where(fpr <= 0.01)[0][-1]] * 100

print(f"\n===== Verification Results =====")
print(f"EER: {eer:.2f}%")
print(f"TAR @ 1% FAR: {tar_at_1_far:.2f}%")

# ------------------ Identification evaluation ------------------ #

print("\nRunning speaker identification evaluation...")
correct = 0
total = 0

for speaker_id, embeddings in tqdm(speaker_embeddings.items()):
    for emb in embeddings:
        scores = {spk: np.dot(emb, spk_emb) for spk, spk_emb in speaker_avg_embeddings.items()}
        predicted_speaker = max(scores, key=scores.get)
        if predicted_speaker == speaker_id:
            correct += 1
        total += 1

accuracy = correct / total * 100

print(f"\n===== Identification Results =====")
print(f"Speaker Identification Accuracy: {accuracy:.2f}%")
