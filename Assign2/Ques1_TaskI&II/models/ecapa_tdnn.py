import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_base

class TDNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation, padding):
        super().__init__()
        self.tdnn = nn.Conv1d(input_dim, output_dim, context_size,
                              dilation=dilation, padding=padding)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.tdnn(x)
        x = self.activation(x)
        x = self.bn(x)
        return x

class ECAPA_TDNN_SMALL(nn.Module):
    def __init__(self, feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None):
        super().__init__()
        # Use torchaudio pretrained wav2vec2 XLSR as frontend
        self.frontend = wav2vec2_base().feature_extractor # feature extractor only
        # TDNN layers (simplified version)
        self.tdnn1 = TDNNBlock(input_dim=512, output_dim=512, context_size=5, dilation=1, padding=2)
        self.tdnn2 = TDNNBlock(input_dim=512, output_dim=512, context_size=3, dilation=2, padding=2)
        self.tdnn3 = TDNNBlock(input_dim=512, output_dim=512, context_size=3, dilation=3, padding=3)
        self.fc = nn.Linear(512, 192)

    def forward(self, x):
        with torch.no_grad():
            length = torch.tensor([x.shape[1]], device=x.device)
            features, _ = self.frontend(x, length)

        # Transpose to match TDNN expected input: (batch, channels, time)
        features = features.transpose(1, 2)  # from (batch, time, features) to (batch, features, time)

        # TDNN layers
        out = self.tdnn1(features)
        out = self.tdnn2(out)
        out = self.tdnn3(out)

        # Temporal average pooling
        pooled = torch.mean(out, dim=2)

        # Final linear projection
        embeddings = self.fc(pooled)
        return embeddings
