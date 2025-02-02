#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


# Define dataset paths
csv_file = "/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_base_path = "/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/audio"


# In[11]:


# Load the dataset metadata
df = pd.read_csv(csv_file)

# Display first few rows
print("✅ UrbanSound8K metadata loaded successfully!")
print(df.head())  # Show first few rows to verify structure


# In[12]:


# Select a valid audio file from the dataset
selected_file = None

for _, row in df.iterrows():
    file_path = os.path.join(audio_base_path, f"fold{row['fold']}", row["slice_file_name"])
    
    if os.path.exists(file_path):  # Ensure the file exists
        selected_file = file_path
        print(f"✅ Found valid audio file: {selected_file}")
        break

# Raise an error if no valid file is found
if selected_file is None:
    raise FileNotFoundError("❌ No valid audio files found. Check dataset paths.")


# In[13]:


# Load the selected audio file
waveform, sample_rate = torchaudio.load(selected_file)

# Print file info
print(f"✅ Loaded: {selected_file}")
print(f"Sample Rate: {sample_rate}, Waveform Shape: {waveform.shape}")


# In[14]:


# Define STFT parameters
n_fft = 1024  # FFT size
hop_length = 256  # Hop length
win_length = 1024  # Window size

# Define window functions
hann_window = torch.hann_window(win_length)
hamming_window = torch.hamming_window(win_length)
rectangular_window = torch.ones(win_length)  # Rectangular window

# Apply STFT with different windows
stft_hann = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=hann_window, return_complex=True)
stft_hamming = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=hamming_window, return_complex=True)
stft_rectangular = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=rectangular_window, return_complex=True)

print("✅ STFT applied successfully!")


# In[15]:


# Convert to magnitude spectrograms
spectrogram_hann = torch.abs(stft_hann).numpy()
spectrogram_hamming = torch.abs(stft_hamming).numpy()
spectrogram_rectangular = torch.abs(stft_rectangular).numpy()

print("✅ Spectrograms computed successfully!")


# In[24]:


# Function to plot spectrograms
def plot_spectrogram(spectrogram, title, ax):
    ax.imshow(20 * torch.log10(torch.tensor(spectrogram) + 1e-6).numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Plot each spectrogram
plot_spectrogram(spectrogram_hann[0], "Hann Window Spectrogram", axes[0])
plot_spectrogram(spectrogram_hamming[0], "Hamming Window Spectrogram", axes[1])
plot_spectrogram(spectrogram_rectangular[0], "Rectangular Window Spectrogram", axes[2])

plt.tight_layout()
plt.show()


# In[ ]:




