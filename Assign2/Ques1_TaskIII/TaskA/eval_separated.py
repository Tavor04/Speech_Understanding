import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from tqdm import tqdm

# ===========================
# Configuration
# ===========================

ESTIMATED_DIR = './estimated_sources'
DATASET_PATH = './mixed_vox2'
SAMPLE_RATE = 8000

TEST_S1_PATH = os.path.join(DATASET_PATH, 'test', 's1')
TEST_S2_PATH = os.path.join(DATASET_PATH, 'test', 's2')

OUTPUT_CSV = './evaluation_results.csv'

# ===========================
# Device setup
# ===========================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ===========================
# Utility functions
# ===========================

def load_audio(file_path, device=DEVICE):
    """Load an audio file and move it to the correct device."""
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch! Expected {SAMPLE_RATE}, got {sr} in {file_path}")
    waveform = waveform.to(device)
    return waveform.squeeze(0)  # [time]

def compute_metrics(est_sources, true_sources):
    # Align lengths
    min_length = min(est_sources.shape[1], true_sources.shape[1])
    est_sources = est_sources[:, :min_length]
    true_sources = true_sources[:, :min_length]

    # Move to numpy
    est_sources_np = est_sources.detach().cpu().numpy()
    true_sources_np = true_sources.detach().cpu().numpy()

    # Compute SDR, SIR, SAR
    sdr, sir, sar, _ = bss_eval_sources(true_sources_np, est_sources_np, compute_permutation=False)

    # Compute PESQ
    pesq_scores = []
    for i in range(true_sources_np.shape[0]):
        pesq_score = pesq(SAMPLE_RATE, true_sources_np[i], est_sources_np[i], 'nb')
        pesq_scores.append(pesq_score)

    return sdr, sir, sar, pesq_scores

# ===========================
# Evaluation loop
# ===========================

print("[INFO] Starting evaluation...")
est_files = sorted(f for f in os.listdir(ESTIMATED_DIR) if f.endswith('_source1.wav'))

# Store results
results = []

for file in tqdm(est_files, desc="Evaluating files", ncols=80):
    try:
        file_base = file.replace('_source1.wav', '')

        # Paths
        est_s1_path = os.path.join(ESTIMATED_DIR, f"{file_base}_source1.wav")
        est_s2_path = os.path.join(ESTIMATED_DIR, f"{file_base}_source2.wav")
        true_s1_path = os.path.join(TEST_S1_PATH, f"{file_base}.wav")
        true_s2_path = os.path.join(TEST_S2_PATH, f"{file_base}.wav")

        # Load audio
        est_s1 = load_audio(est_s1_path)
        est_s2 = load_audio(est_s2_path)
        true_s1 = load_audio(true_s1_path)
        true_s2 = load_audio(true_s2_path)

        # Stack sources
        est_sources = torch.stack([est_s1, est_s2], dim=0)  # [2, time]
        true_sources = torch.stack([true_s1, true_s2], dim=0)  # [2, time]

        def normalize_audio(waveform):
            return waveform / (waveform.abs().max() + 1e-8)
        
        est_sources = normalize_audio(est_sources)
        true_sources = normalize_audio(true_sources)

        # Compute metrics
        sdr, sir, sar, pesq_scores = compute_metrics(est_sources, true_sources)

        # Store results
        results.append({
            'file': file_base,
            'SDR_source1': sdr[0],
            'SDR_source2': sdr[1],
            'SIR_source1': sir[0],
            'SIR_source2': sir[1],
            'SAR_source1': sar[0],
            'SAR_source2': sar[1],
            'PESQ_source1': pesq_scores[0],
            'PESQ_source2': pesq_scores[1],
        })

    except Exception as e:
        print(f"[ERROR] Failed processing {file}: {e}")

# ===========================
# Results summary
# ===========================

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Results saved to {OUTPUT_CSV}")

# Print average metrics
if not df.empty:
    print("\n===== Average Evaluation Results =====")
    for metric in ['SDR', 'SIR', 'SAR', 'PESQ']:
        metric_cols = [col for col in df.columns if col.startswith(metric)]
        avg_value = df[metric_cols].mean().mean()
        print(f"{metric}: {avg_value:.2f}")
else:
    print("[WARNING] No results to summarize!")

print("[INFO] Evaluation complete")
