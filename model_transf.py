import torch
import torch.nn as nn

class EnergyCorrelatorClassifier(nn.Module):
    def __init__(self, num_angles, num_states, num_classes, theta_embedding_dim, state_embedding_dim, hidden_dim, num_heads, num_layers):
        super(EnergyCorrelatorClassifier, self).__init__()

        # Angular and state embeddings
        self.theta_embedding = nn.Embedding(num_angles, theta_embedding_dim)
        self.state_embedding = nn.Embedding(num_states, state_embedding_dim)
        
        # Special classification token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, theta_embedding_dim))  

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=theta_embedding_dim,  # Model dimension = embedding size
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected classifier
        self.fc = nn.Linear(theta_embedding_dim, num_classes)

    def forward(self, theta_indices, state_indices):
        batch_size = theta_indices.shape[0]

        # Get embeddings
        theta_embedded = self.theta_embedding(theta_indices)  # (batch, seq_len, embed_dim)
        state_embedded = self.state_embedding(state_indices)  # (batch, embed_dim)

        # Optionally, add state embeddings
        combined = theta_embedded + state_embedded.unsqueeze(1)  # (batch, seq_len, embed_dim)

        # Add [CLS] token at the end of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        combined = torch.cat([combined, cls_tokens], dim=1)  # (batch, seq_len+1, embed_dim)

        # Transformer encoder
        encoded = self.transformer(combined)  # (batch, seq_len+1, embed_dim)

        # Use the CLS token output for classification
        cls_output = encoded[:, -1, :]  # Take the last token (CLS)
        output = self.fc(cls_output)  # (batch, num_classes)

        return output

