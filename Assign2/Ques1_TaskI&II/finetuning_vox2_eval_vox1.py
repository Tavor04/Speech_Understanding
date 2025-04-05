import os
import random
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
from collections import defaultdict
import loralib as lora
from datetime import datetime

# ------------------ Config ------------------ #
TRAIN_DATASET = './vox2_converted_wav'
VOX1_DATASET = './vox1_test_wav'
VERI_PAIRS = './veri_test2.txt'
CHECKPOINT_PATH = './wav2vec2_xlsr_SV_fixed.th'
SAVE_CHECKPOINT = './finetuned_lora_model.pth'
RESULTS_FILE = './training_results.txt'

BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# ------------------ Data preparation ------------------ #
def get_speakers(data_path):
    speakers = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return speakers

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Force to mono (channel first)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Mono audio, ensure shape [1, time]
    elif waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Stereo to mono

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(DEVICE)
        waveform = resampler(waveform)

    return waveform.float().to(DEVICE)

def create_dataset(data_path, speakers):
    dataset = []
    for spk in speakers:
        spk_folder = os.path.join(data_path, spk)
        for utt_folder in os.listdir(spk_folder):
            utt_folder_path = os.path.join(spk_folder, utt_folder)
            if not os.path.isdir(utt_folder_path):
                continue
            for file in os.listdir(utt_folder_path):
                if file.endswith('.wav'):
                    dataset.append((os.path.join(utt_folder_path, file), spk))
    return dataset

# ------------------ Model & Loss ------------------ #
from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # Use your existing ecapa_tdnn.py

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = F.one_hot(label, num_classes=cosine.size(1)).float().to(DEVICE)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# ------------------ Apply LoRA ------------------ #
def apply_lora_to_linear_layers(model, rank=4, alpha=32):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_linear = lora.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=rank,
                lora_alpha=alpha,
                bias=module.bias is not None
            )
            lora_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_linear.bias.data = module.bias.data.clone()

            parent_module = model
            sub_names = name.split('.')
            for sub_name in sub_names[:-1]:
                parent_module = getattr(parent_module, sub_name)
            setattr(parent_module, sub_names[-1], lora_linear)

# ------------------ Training loop ------------------ #
def train(model, classifier, dataset, spk2id, optimizer, criterion, epoch):
    model.train()
    classifier.train()
    random.shuffle(dataset)
    total_batches = len(dataset) // BATCH_SIZE + int(len(dataset) % BATCH_SIZE != 0)
    total_loss = 0
    with tqdm(total=total_batches, desc=f"Epoch {epoch+1} Training") as pbar:
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i + BATCH_SIZE]
            waveforms = []
            labels = []
            for file, spk in batch:
                waveform = load_audio(file)  # Shape: [1, time]
                waveform = waveform.squeeze(0)  # Shape: [time]
                waveforms.append(waveform)
                labels.append(spk2id[spk])

            # Pad the time dimension
            padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)  # Shape: [batch_size, max_time]
            waveforms = padded_waveforms.unsqueeze(1)  # Shape: [batch_size, 1, max_time]
            waveforms = waveforms.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(waveforms.squeeze(1))  # Shape: [batch_size, 192]
            logits = classifier(embeddings, labels)  # Shape: [batch_size, num_classes]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)

    avg_loss = total_loss / total_batches
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

# ------------------ Embedding extraction ------------------ #
def extract_embeddings(model, dataset, description):
    model.eval()
    embeddings = defaultdict(list)
    with torch.no_grad():
        for file, spk in tqdm(dataset, desc=description):
            waveform = load_audio(file)
            waveform = waveform.squeeze(0)
            waveform = waveform.unsqueeze(0)  # Shape: [1, time]
            embedding = model(waveform)
            embeddings[spk].append(embedding.squeeze(0).cpu().numpy())
    avg_embeddings = {spk: np.mean(embs, axis=0) for spk, embs in embeddings.items()}
    return avg_embeddings

# ------------------ Evaluation ------------------ #
def evaluate_identification(model, dataset, reference_embeddings, description):
    correct = 0
    total = 0
    with torch.no_grad():
        for file, true_spk in tqdm(dataset, desc=description):
            waveform = load_audio(file)
            waveform = waveform.squeeze(0) 
            waveform = waveform.unsqueeze(0)  # Shape: [1, time]
            embedding = model(waveform).squeeze(0).cpu().numpy()
            scores = {spk: np.dot(embedding, ref_emb) for spk, ref_emb in reference_embeddings.items()}
            predicted_spk = max(scores, key=scores.get)
            if predicted_spk == true_spk:
                correct += 1
            total += 1
    accuracy = correct / total * 100
    print(f"{description} Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_verification(model, veri_pairs, reference_embeddings):
    scores = []
    labels = []
    with open(veri_pairs, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Verification Evaluation"):
        label, path1, path2 = line.strip().split()
        spk1 = path1.split('/')[0]
        spk2 = path2.split('/')[0]
        if spk1 not in reference_embeddings or spk2 not in reference_embeddings:
            continue
        emb1 = reference_embeddings[spk1]
        emb2 = reference_embeddings[spk2]
        score = np.dot(emb1, emb2)
        scores.append(score)
        labels.append(int(label))

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] * 100
    tar_at_1_far = tpr[np.where(fpr <= 0.01)[0][-1]] * 100

    print(f"EER: {eer:.2f}%")
    print(f"TAR @ 1% FAR: {tar_at_1_far:.2f}%")
    return eer, tar_at_1_far

# ------------------ Main ------------------ #
if __name__ == "__main__":
    TRAIN_IF_NEEDED = False
    # Prepare speakers
    speakers = get_speakers(TRAIN_DATASET)
    train_speakers = speakers[:100]
    test_speakers = speakers[100:]

    train_dataset = create_dataset(TRAIN_DATASET, train_speakers)
    test_dataset = create_dataset(TRAIN_DATASET, test_speakers)
    vox1_speakers = get_speakers(VOX1_DATASET)
    vox1_dataset = create_dataset(VOX1_DATASET, vox1_speakers)

    spk2id = {spk: idx for idx, spk in enumerate(train_speakers)}

    # Load model and checkpoint
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Apply LoRA
    apply_lora_to_linear_layers(model)
    model = model.to(DEVICE)
    # Ensure all weights are trainable (frontend, backend, and classifier)
    for param in model.parameters():
        param.requires_grad = True

    # ArcFace classifier
    classifier = ArcMarginProduct(192, len(spk2id)).to(DEVICE)
    for param in classifier.parameters():
        param.requires_grad = True

    if os.path.exists(SAVE_CHECKPOINT) and not TRAIN_IF_NEEDED:
        print(f"\n✅ Finetuned checkpoint found at {SAVE_CHECKPOINT}, skipping training.")
        finetuned_ckpt = torch.load(SAVE_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(finetuned_ckpt['model'], strict=False)
        classifier.load_state_dict(finetuned_ckpt['classifier'], strict=False)
    else:
        print("\n🚀 Starting training...")

        # ✅ Optimizer and loss function should go here:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=LR)
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(EPOCHS):
            train(model, classifier, train_dataset, spk2id, optimizer, criterion, epoch)
    
        torch.save({'model': model.state_dict(), 'classifier': classifier.state_dict()}, SAVE_CHECKPOINT)
        print(f"\n✅ Checkpoint saved to {SAVE_CHECKPOINT}")

    # Evaluation
    print("\nEvaluating on VoxCeleb2 (Test set)...")
    test_embeddings = extract_embeddings(model, test_dataset, "Test Set Embeddings")
    vox2_test_acc = evaluate_identification(model, test_dataset, test_embeddings, "VoxCeleb2 Identification")

    print("\nEvaluating on VoxCeleb1...")
    vox1_embeddings = extract_embeddings(model, vox1_dataset, "VoxCeleb1 Embeddings")
    eer, tar = evaluate_verification(model, VERI_PAIRS, vox1_embeddings)
    vox1_id_acc = evaluate_identification(model, vox1_dataset, vox1_embeddings, "VoxCeleb1 Identification")

    # Save results to file
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n")
        f.write(f"VoxCeleb2 Test Identification Accuracy: {vox2_test_acc:.2f}%\n")
        f.write(f"VoxCeleb1 Verification EER: {eer:.2f}%\n")
        f.write(f"VoxCeleb1 Verification TAR@1%FAR: {tar:.2f}%\n")
        f.write(f"VoxCeleb1 Identification Accuracy: {vox1_id_acc:.2f}%\n")

    print(f"\nResults saved to {RESULTS_FILE}")