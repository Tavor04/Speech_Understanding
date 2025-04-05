import os
import sys
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# Configuration
# ===========================

SAMPLE_RATE = 16000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
ESTIMATED_DIR = './estimated_sources'
REFERENCE_DIR = './vox2_test_aac'
MIXED_TEST_S1 = './mixed_vox2/test/s1'
MIXED_TEST_S2 = './mixed_vox2/test/s2'

PRETRAINED_CHECKPOINT = './wav2vec2_xlsr_SV_fixed.th'
FINETUNED_CHECKPOINT = './finetuned_lora_model.pth'

print(f"[INFO] Using device: {DEVICE}")

# ===========================
# Import your model properly
# ===========================

sys.path.append('./Ques1')
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

# ===========================
# Utility functions
# ===========================

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.to(DEVICE)

def get_speaker_ids(path, count):
    return sorted(os.listdir(path))[:count]


def extract_embedding(model, waveform):
    with torch.no_grad():
        waveform = waveform.squeeze(0)      # [time]
        waveform = waveform.unsqueeze(0)    # [1, time]
        embedding = model(waveform)
    return embedding.squeeze(0).cpu().numpy()


def compute_rank1_accuracy(estimated_embeddings, estimated_labels, reference_embeddings, reference_labels):
    correct = 0
    total = len(estimated_embeddings)

    similarities = cosine_similarity(np.stack(estimated_embeddings), np.stack(reference_embeddings))

    for idx, sim_vector in enumerate(similarities):
        predicted_index = np.argmax(sim_vector)
        if reference_labels[predicted_index] == estimated_labels[idx]:
            correct += 1

    return correct / total if total > 0 else 0

# ===========================
# Load Models (your way)
# ===========================

print("[INFO] Loading models...")

# Pre-trained model
pretrained_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None).to(DEVICE).eval()
pretrained_checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE)
pretrained_model.load_state_dict(pretrained_checkpoint['model'], strict=False)

# Fine-tuned model
finetuned_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None).to(DEVICE).eval()
finetuned_checkpoint = torch.load(FINETUNED_CHECKPOINT, map_location=DEVICE)
finetuned_model.load_state_dict(finetuned_checkpoint['model'], strict=False)

print("[INFO] Models loaded successfully.")

# ===========================
# Load reference embeddings
# ===========================

print("[INFO] Loading reference embeddings...")

reference_embeddings_pretrained = []
reference_embeddings_finetuned = []
reference_labels = []

all_speakers = get_speaker_ids(REFERENCE_DIR, 100)
test_speakers = all_speakers[50:100]

for speaker_id in tqdm(test_speakers, desc="Reference speakers", ncols=80):
    speaker_dir = os.path.join(REFERENCE_DIR, speaker_id)
    for root, _, files in os.walk(speaker_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                waveform = load_audio(filepath)

                emb_pretrained = extract_embedding(pretrained_model, waveform)
                emb_finetuned = extract_embedding(finetuned_model, waveform)

                reference_embeddings_pretrained.append(emb_pretrained)
                reference_embeddings_finetuned.append(emb_finetuned)
                reference_labels.append(speaker_id)

print(f"[INFO] Loaded {len(reference_labels)} reference embeddings.")

# ===========================
# Load estimated embeddings
# ===========================

print("[INFO] Loading estimated source embeddings...")

estimated_embeddings_pretrained = []
estimated_embeddings_finetuned = []
estimated_labels = []

estimated_files = sorted(f for f in os.listdir(ESTIMATED_DIR) if f.endswith('.wav'))

for file in tqdm(estimated_files, desc="Estimated sources", ncols=80):
    file_base, source_type = file.replace('.wav', '').split('_')

    # Determine speaker label from mixed clean source
    true_source_path = os.path.join(
        MIXED_TEST_S1 if source_type == 'source1' else MIXED_TEST_S2,
        f"{file_base}.wav"
    )

    if not os.path.exists(true_source_path):
        continue  # skip if missing

    true_base = os.path.basename(true_source_path).replace('.wav', '')

    speaker_found = False
    original_speaker_id = None

    for speaker_id in test_speakers:
        speaker_dir = os.path.join(REFERENCE_DIR, speaker_id)
        for root, _, files in os.walk(speaker_dir):
            for ref_file in files:
                if ref_file.replace('.wav', '') == true_base:
                    original_speaker_id = speaker_id
                    speaker_found = True
                    break
            if speaker_found:
                break
        if speaker_found:
            break

    if not speaker_found or original_speaker_id is None:
        print(f"[WARNING] Speaker not found for {file}")
        continue

    # Extract embedding
    est_waveform = load_audio(os.path.join(ESTIMATED_DIR, file))
    emb_pretrained = extract_embedding(pretrained_model, est_waveform)
    emb_finetuned = extract_embedding(finetuned_model, est_waveform)

    estimated_embeddings_pretrained.append(emb_pretrained)
    estimated_embeddings_finetuned.append(emb_finetuned)
    estimated_labels.append(original_speaker_id)

print(f"[INFO] Loaded {len(estimated_labels)} estimated embeddings.")

# ===========================
# Compute Rank-1 Accuracy
# ===========================

print("[INFO] Computing Rank-1 accuracy...")

rank1_pretrained = compute_rank1_accuracy(
    estimated_embeddings_pretrained,
    estimated_labels,
    reference_embeddings_pretrained,
    reference_labels
)

rank1_finetuned = compute_rank1_accuracy(
    estimated_embeddings_finetuned,
    estimated_labels,
    reference_embeddings_finetuned,
    reference_labels
)

print("\n===== Rank-1 Identification Results =====")
print(f"Pre-trained model Rank-1 accuracy: {rank1_pretrained * 100:.2f}%")
print(f"Fine-tuned model Rank-1 accuracy: {rank1_finetuned * 100:.2f}%")
print("[INFO] Evaluation complete")
