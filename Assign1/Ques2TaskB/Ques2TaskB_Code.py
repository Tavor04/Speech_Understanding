#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os

# Define file paths
audio_files = [
    "/Users/malik/Music/rekordbox/Music for DJing/Becky G, Maluma - La Respuesta (Official Video).mp3",
    "/Users/malik/Music/rekordbox/Music for DJing/Guns N' Roses - Paradise City (Official Music Video).mp3",
    "/Users/malik/Music/rekordbox/Music for DJing/50 Cent - Candy Shop (Official Music Video) ft. Olivia.mp3",
    "/Users/malik/Music/rekordbox/Music for DJing/Martin Garrix - Animals (Official Video).mp3"
]

# Spectrogram parameters
sample_rate = 22050  
n_fft = 1024
hop_length = 512

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):  # Avoid reconversion
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
    return wav_path

# To generate and plot spectrogram
def plot_spectrogram(audio_path, ax, title):
    # Convert MP3 to WAV
    if audio_path.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)
    
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        waveform = T.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    
    # Compute Spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
    
    # Convert to decibels
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    
    # Plot spectrogram
    ax.imshow(spectrogram_db.squeeze().numpy(), aspect='auto', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency Bins")

# Plot all spectrograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

titles = ["Blues", "Rock", "Hip-Hop", "Electronic"]
for i, (file, ax, title) in enumerate(zip(audio_files, axes.flatten(), titles)):
    try:
        plot_spectrogram(file, ax, title)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        ax.set_title(f"Error loading {title}")

plt.tight_layout()
plt.show()


# In[ ]:




