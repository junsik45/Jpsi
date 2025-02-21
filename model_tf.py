import torch
import torch.nn as nn
import math

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, num_classes, d_model=512, num_heads=4, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.seq_len = seq_len  # Sequence length (input_dim)

        # Input projection (each number in the sequence is treated as a token)
        self.embedding = nn.Linear(1, d_model)  # Maps each scalar number to a d_model vector

        # Positional Encoding
        self.positional_encoding = self.create_positional_encoding(seq_len, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensures (batch, seq_len, feature_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Linear(d_model, num_classes)

    def create_positional_encoding(self, seq_len, d_model):
        """Creates sinusoidal positional encoding for any sequence length."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape  # x is (batch_size, seq_len)

        # Reshape x to (batch_size, seq_len, 1) so that each number is treated as a token
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Project input numbers into d_model-dimensional space
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # Ensure positional encoding matches input size
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc  # Add positional encoding (broadcasting over batch)

        # Transformer Encoder
        x = self.transformer(x)  # (batch_size, seq_len, d_model)

        # Classification: Use the first token's representation (like CLS token in BERT)
        x = x[:, 0, :]  # Take the first token for classification

        return self.fc(x)  # (batch_size, num_classes)

