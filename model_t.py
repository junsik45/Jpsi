class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Project input to hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # Classification token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, num_classes)  # Classification head

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)

        # Append CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_dim)

        x = self.transformer(x)  # Transformer processing
        cls_output = x[:, 0, :]  # Use CLS token output
        return self.fc(cls_output)

