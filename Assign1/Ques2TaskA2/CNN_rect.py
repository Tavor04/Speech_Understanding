#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torchaudio
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# In[2]:


import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

class UrbanSoundDatasetRectangular(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 512
        self.max_frames = 200  # Fixed size for spectrograms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fold_folder = f"fold{row['fold']}"
        file_path = os.path.join(self.root_dir, fold_folder, row['slice_file_name'])
        label = row['classID']

        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            return torch.zeros((1, self.n_fft // 2 + 1, self.max_frames)), torch.tensor(-1, dtype=torch.long)

        try:
            # Load audio file
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:  # Convert stereo to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to a fixed sample rate
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

            # ✅ Define Rectangular Window (all ones)
            rect_window = torch.ones(self.n_fft)  # Rectangular window

            # ✅ Apply Rectangular Window & Compute Spectrogram
            spectrogram = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=rect_window,  # ✅ Fix: Use rectangular window
                return_complex=False
            )

            # Convert complex STFT output to real magnitude spectrogram
            spectrogram = spectrogram.pow(2).sum(-1).sqrt()

            # Ensure correct shape: [1, frequency_bins, time_frames]
            spectrogram = spectrogram.squeeze(0)  # Remove unnecessary dimension
            spectrogram = spectrogram.unsqueeze(0)  # Ensure it is [1, freq_bins, time_frames]

            # Ensure Fixed Size (Padding or Truncation)
            time_dim = spectrogram.shape[-1]
            if time_dim < self.max_frames:
                pad = self.max_frames - time_dim
                spectrogram = torch.nn.functional.pad(spectrogram, (0, pad), "constant", 0)
            elif time_dim > self.max_frames:
                spectrogram = spectrogram[:, :, :self.max_frames]

            return spectrogram, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, self.n_fft // 2 + 1, self.max_frames)), torch.tensor(-1, dtype=torch.long)


# In[3]:


batch_size = 32
epochs = 10
learning_rate = 0.001
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# In[5]:


root_dir = "/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/audio/"
dataset = UrbanSoundDatasetRectangular("/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/metadata/UrbanSound8K.csv", root_dir)

def collate_fn(batch):
    batch = [b for b in batch if b[1] != -1]
    return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.empty(0), torch.empty(0))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# In[8]:


from sklearn.model_selection import train_test_split

# Load full dataset
full_dataset = UrbanSoundDatasetRectangular(
    csv_file="/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/metadata/UrbanSound8K.csv",
    root_dir="/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/audio/"
)

# Get indices for train and test splits
train_indices, test_indices = train_test_split(
    list(range(len(full_dataset))), test_size=0.2, random_state=42, stratify=full_dataset.data["classID"]
)

# Create subsets for training and testing
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# In[9]:


import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(64*1*1, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AudioCNN().to(device)


# In[10]:


model = AudioCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[11]:


# Training Loop
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:  # Use train_loader here
        if inputs.shape[0] == 0:
            continue
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")


# In[12]:


# Evaluate Model on Test Set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:  # Use test_loader here
        if inputs.shape[0] == 0:
            continue
        
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")


# In[ ]:




