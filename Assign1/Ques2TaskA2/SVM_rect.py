#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.functional as F


# In[2]:


# Define dataset paths
csv_file = "/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_base_path = "/Users/malik/Downloads/Projects/Speech Understanding Projects/windowing/UrbanSound8K/audio"


# In[3]:


# Load the dataset metadata
df = pd.read_csv(csv_file)

# Display first few rows
print("✅ UrbanSound8K metadata loaded successfully!")
print(df.head())  # Show first few rows to verify structure


# In[9]:


import torch
import torchaudio
import torchaudio.functional as TAF

def extract_spectral_features_rectangular(audio_path):
    try:
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)

        # Convert to single channel if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Define STFT parameters
        n_fft = 1024
        hop_length = 256
        win_length = 1024
        window = torch.ones(win_length)  # Rectangular window

        # Compute STFT
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

        # Compute magnitude spectrogram
        spectrogram = torch.abs(stft)
        power_spectrogram = spectrogram ** 2

        # Compute Spectral Features
        spectral_centroid = TAF.spectral_centroid(
            waveform, sample_rate=sr, pad=0, window=window, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        ).mean()
        
        freqs = torch.linspace(0, sr // 2, steps=(n_fft // 2) + 1)
        spectral_bandwidth = torch.sqrt(torch.sum(((freqs - spectral_centroid) ** 2) * spectrogram.mean(dim=-1))).mean()

        geometric_mean = torch.exp(torch.mean(torch.log(power_spectrogram + 1e-6), dim=-1))
        arithmetic_mean = torch.mean(power_spectrogram, dim=-1)
        spectral_flatness = (geometric_mean / arithmetic_mean).mean()

        zero_crossing_rate = (torch.diff(torch.sign(waveform)) != 0).float().mean()

        # Combine features into a tensor
        features = torch.tensor([
            spectral_centroid,
            spectral_bandwidth,
            spectral_flatness,
            zero_crossing_rate
        ])

        return features.numpy()

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


# In[10]:


X = []  # Feature matrix
y = []  # Labels

# Loop through all rows in the dataset
for _, row in df.iterrows():
    # Construct full file path
    file_path = os.path.join(audio_base_path, f"fold{row['fold']}", row["slice_file_name"])

    if os.path.exists(file_path):  # Ensure the file exists
        features = extract_spectral_features_rectangular(file_path)
        if features is not None:  # Ensure successful feature extraction
            X.append(features)
            y.append(row["classID"])  # Store class label

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

print("✅ Feature extraction complete!")
print(f"Feature Matrix Shape: {X.shape}")
print(f"Number of Labels: {len(y)}")


# In[11]:


# Check for NaN values
nan_count = np.isnan(X).sum()
print(f"Total NaN values in X: {nan_count}")


# In[12]:


# Replace NaNs with column means
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)


# In[13]:


print(f"Total NaN values in X: {np.isnan(X).sum()}")


# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Convert lists to NumPy arrays
X = np.array(X)  # Feature matrix
y = np.array(y)  # Labels (class IDs)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("✅ Data prepared for training")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# In[15]:


from sklearn.svm import SVC

# Train an SVM classifier
svm_classifier = SVC(kernel="rbf", C=1.0, gamma="scale")  # Using RBF kernel
svm_classifier.fit(X_train, y_train)

print("✅ SVM model trained successfully!")


# In[16]:


from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred_svm = svm_classifier.predict(X_test)

# Compute accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"✅ SVM Classifier Accuracy: {accuracy_svm:.2f}")

# Generate classification report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred_svm))



# In[ ]:




