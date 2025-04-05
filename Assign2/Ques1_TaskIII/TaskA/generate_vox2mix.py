import os
import torchaudio
import random
import torch
from glob import glob
from tqdm import tqdm

random.seed(42)

AAC_ROOT = "vox2_test_aac"
OUTPUT_DIR = "mixed_vox2"
SAMPLE_RATE = 16000
TARGET_MIXES = 100
MIN_DURATION = 3.0  # in seconds

def get_speaker_ids(path, count):
    return sorted(os.listdir(path))[:count]

def convert_to_wav(m4a_path):
    wav_path = m4a_path.replace(".m4a", ".wav")
    if not os.path.exists(wav_path):
        audio, sr = torchaudio.load(m4a_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
        audio = audio.mean(dim=0, keepdim=True)  # mono
        torchaudio.save(wav_path, audio, SAMPLE_RATE)
    return wav_path

def get_all_wavs_for_speaker(speaker_path):
    m4a_files = glob(os.path.join(speaker_path, "*", "*.m4a"))
    wav_files = [convert_to_wav(path) for path in m4a_files]
    return wav_files

def pad_signals(sig1, sig2):
    max_len = max(sig1.shape[-1], sig2.shape[-1])
    sig1 = torch.nn.functional.pad(sig1, (0, max_len - sig1.shape[-1]))
    sig2 = torch.nn.functional.pad(sig2, (0, max_len - sig2.shape[-1]))
    return sig1, sig2

def mix_and_save(pair_id, sig1, sig2, out_root):
    sig1, sig2 = pad_signals(sig1, sig2)
    mix = sig1 + sig2
    mix = mix / mix.abs().max()

    torchaudio.save(os.path.join(out_root, "mix_clean", f"{pair_id}.wav"), mix, SAMPLE_RATE)
    torchaudio.save(os.path.join(out_root, "s1", f"{pair_id}.wav"), sig1, SAMPLE_RATE)
    torchaudio.save(os.path.join(out_root, "s2", f"{pair_id}.wav"), sig2, SAMPLE_RATE)

def ensure_dirs(out_root):
    os.makedirs(os.path.join(out_root, "mix_clean"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "s1"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "s2"), exist_ok=True)

def generate_mixture_set(speaker_ids, out_root):
    ensure_dirs(out_root)
    used_pairs = set()
    total_speakers = len(speaker_ids)
    pair_id = 0

    print(f"\n Generating {TARGET_MIXES} mixtures in {out_root}")
    pbar = tqdm(total=TARGET_MIXES, desc="Mixes created")

    while pair_id < TARGET_MIXES:
        spk1, spk2 = random.sample(speaker_ids, 2)
        if (spk1, spk2) in used_pairs or (spk2, spk1) in used_pairs:
            continue

        spk1_path = os.path.join(AAC_ROOT, spk1)
        spk2_path = os.path.join(AAC_ROOT, spk2)

        files1 = get_all_wavs_for_speaker(spk1_path)
        files2 = get_all_wavs_for_speaker(spk2_path)

        if not files1 or not files2:
            continue

        random.shuffle(files1)
        random.shuffle(files2)

        # Try all combinations until we find one that works
        success = False
        for f1 in files1:
            for f2 in files2:
                try:
                    sig1, sr1 = torchaudio.load(f1)
                    sig2, sr2 = torchaudio.load(f2)
                    if sig1.shape[1] < SAMPLE_RATE * MIN_DURATION or sig2.shape[1] < SAMPLE_RATE * MIN_DURATION:
                        continue
                    mix_and_save(f"{pair_id:04d}", sig1, sig2, out_root)
                    used_pairs.add((spk1, spk2))
                    pair_id += 1
                    pbar.update(1)
                    success = True
                    break
                except Exception:
                    continue
            if success:
                break

    pbar.close()

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    all_speakers = get_speaker_ids(AAC_ROOT, 100)
    train_speakers = all_speakers[:50]
    test_speakers = all_speakers[50:100]

    generate_mixture_set(train_speakers, os.path.join(OUTPUT_DIR, "train"))
    generate_mixture_set(test_speakers, os.path.join(OUTPUT_DIR, "test"))

    print("\n Guaranteed mixing complete. Output saved in:", OUTPUT_DIR)
