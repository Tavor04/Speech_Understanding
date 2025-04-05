
# 🎙️ Speech Understanding: Speaker Separation, Identification, and Language Classification

## Overview

This project addresses multiple speech understanding tasks as part of Assignment 2:
- **Speaker Verification** using pretrained and fine-tuned models
- **Speech Separation & Enhancement** using SepFormer
- **Rank-1 Speaker Identification** post-separation
- **Language Identification** using MFCC features and SVM classifier

The pipeline spans from dataset preparation, model training, evaluation, to final reporting of metrics including **SDR, SIR, SAR, PESQ**, **EER, TAR**, **Rank-1 accuracy**, and **Language classification accuracy**.

---

## Project Pipeline

### Question 1: Speech Enhancement & Speaker Identification

#### I. Dataset Download and Preparation
- **VoxCeleb1**: Saved as `vox1_test_wav` (for evaluation).
- **VoxCeleb2**: Saved as `vox2_test_aac`, converted to WAV.
- **Splits**:
  - First 50 identities for training.
  - Next 50 identities for testing (used for separation + identification).

#### II. Speaker Verification using Pretrained Model and Fine-tuning
- **Model**: ECAPA-TDNN-SMALL (`wav2vec2_xlsr_SV_fixed.th`)
- **Fine-tuning**:
  - Applied **LoRA** and **ArcFace loss**.
  - Trained for 5 epochs on VoxCeleb2.

- **Results**:
  | Metric                        | Pre-trained | Fine-tuned |
  |------------------------------|-------------|------------|
  | EER (%)                      | 55.01       | 12.29      |
  | TAR@1%FAR (%)                | 0.09        | 21.89      |
  | Speaker Identification (%)   | 3.28        | 33.91      |
  | VoxCeleb2 Test ID Accuracy   | -           | 27.69      |

#### III. Multi-Speaker Scenario Dataset Creation and Separation

- **Mixture generation**: `generate_vox2mix.py` script used to create multi-speaker mixtures.
- **Speaker separation & enhancement**: SepFormer (`sepformer-whamr`)
- **Separation evaluation**:
  - **SIR**: 1.20
  - **SAR**: 6.09
  - **SDR**: -6.89
  - **PESQ**: 1.59

#### B. Rank-1 Identification Accuracy After Separation

- **Post-separation Evaluation**:
  - Pre-trained model Rank-1 accuracy: **42.7%**
  - Fine-tuned model Rank-1 accuracy: **91.3%**

- **Conclusion**:
  Fine-tuned model significantly improves identification accuracy, validating the effectiveness of adaptation in multi-speaker scenarios.

---

### Question 2: MFCC Feature Extraction and Language Identification

#### Task A. MFCC Feature Extraction and Comparison
- Dataset: [Indian Languages Audio Dataset (Kaggle)](https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages)
- MFCC Extraction:
  - Used `librosa` to extract MFCC features.
  - Spectrogram visualizations generated.
- Statistical analysis:
  - Mean and variance of MFCC coefficients per language.
  - Insights into phonetic and acoustic diversity across languages.

#### Task B. Classification using MFCC Features
- Model: **Support Vector Machine (SVM)**
- Preprocessing:
  - Feature normalization
  - Train-test split (80/20)
- Results:
  - Classification Accuracy: **73.33%**
  - Confusion matrix generated for analysis.

---

## Project Structure

```
├── generate_vox2mix.py         # Dataset creation script
├── estimated_sources/          # Output from SepFormer
├── mixed_vox2/                 # Multi-speaker mixtures
├── vox2_test_aac/              # Reference speaker dataset
├── models/
│   └── ecapa_tdnn.py           # ECAPA-TDNN-SMALL model
├── rank1accuracy.py            # Rank-1 evaluation script
├── eval_sepformer.py           # Separation quality evaluation
├── wav2vec2_xlsr_SV_fixed.th   # Pre-trained checkpoint
├── finetuned_lora_model.pth    # Fine-tuned checkpoint
├── language_classification.py  # Language identification using MFCC
└── README.md                   # Project documentation
```

---

## How to Run

### 1. Prepare Dataset
```bash
python3 generate_vox2mix.py
```

### 2. Perform Speech Separation
(_Estimated sources already generated and saved in `estimated_sources/`._)

### 3. Evaluate Separation Quality
```bash
python3 eval_sepformer.py
```

### 4. Evaluate Speaker Identification
```bash
python3 rank1accuracy.py
```

### 5. Perform Language Identification (Optional)
```bash
python3 language_classification.py
```

---

## Dependencies

```bash
pip install torch torchaudio librosa scikit-learn tqdm pandas mir_eval pesq matplotlib
```

---

## References

### 📦 Models & Datasets
- [UniSpeech Speaker Verification Repository](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification)
- [SepFormer Pretrained Model](https://huggingface.co/speechbrain/sepformer-whamr)
- [VoxCeleb Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [SpeechBrain ECAPA-TDNN Model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [Indian Languages Audio Dataset (Kaggle)](https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages)

### 🛠️ Evaluation Tools
- [mir_eval](https://github.com/sigsep/sigsep-mus-eval)
- [PESQ Metric](https://github.com/ludlows/python-pesq)
- [tqdm](https://github.com/tqdm/tqdm)

### 📚 Libraries
- PyTorch, torchaudio, librosa, scikit-learn, matplotlib, numpy, pandas
- [SpeechBrain Toolkit](https://github.com/speechbrain/speechbrain)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [ITU-T PESQ Standard](https://www.audiolabs-erlangen.de/resources/QualityMeasure/PESQ)
- [Python Official Documentation](https://www.python.org/)

---

🎉 _This project is now ready for experimentation, evaluation, and reporting!_
