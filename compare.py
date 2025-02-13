import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simulated dataset
num_samples = 5000
num_bins = 64  # Angular bins
num_classes = 5  # Process categories

# Generate random data
X_data = np.random.rand(num_samples, num_bins).astype(np.float32)  # Energy profiles
y_data = np.random.randint(0, num_classes, size=(num_samples,))  # Labels

# Convert to tensors
X_tensor = torch.tensor(X_data)
y_tensor = torch.tensor(y_data)

# Dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define models
mlp_model = MLPClassifier(input_dim=num_bins, num_classes=num_classes)
cnn_model = CNNClassifier(input_dim=num_bins, num_classes=num_classes)
transformer_model = TransformerClassifier(input_dim=num_bins, num_classes=num_classes)

# Training function
def train_model(model, train_loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")

# Train all models
print("Training MLP:")
train_model(mlp_model, train_loader)

print("Training CNN:")
train_model(cnn_model, train_loader)

print("Training Transformer:")
train_model(transformer_model, train_loader)

