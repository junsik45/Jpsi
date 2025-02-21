import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import numpy as np
from model_CNN import CNNClassifier
from model_MLP import MLPClassifier
from model_tf import TransformerClassifier
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from collections import Counter 
warnings.simplefilter(action='ignore', category=FutureWarning)

tensor_dict = {}
data_y = []
df = pd.read_parquet('QEC.parquet', engine='pyarrow')
ev = 160000
df = df.loc[df['eventNum'] < ev ].sort_values(by=['eventNum', 'cos_chi_val'])
df = df[df['category'].isin(['(chi_2c)', '(J/psi[3PJ(8)])', '(J/psi[1S0(8)])']) ]
num_classes = len(df['category'].unique())
cat_dict = { j:i for i, j in enumerate(df['category'].unique()) }
num_bins = 256
linspace_bin = np.linspace(-1., 1., num_bins+1)
print(cat_dict)
for event, group in df.groupby("eventNum"):
    group["cos_chi_bin"] = pd.cut(group["cos_chi_val"], bins=linspace_bin)
    binned = group.groupby("cos_chi_bin")["weight_val"].sum().reset_index()
    binned["bin_midpoint"] = binned["cos_chi_bin"].apply(lambda x: (x.left + x.right) / 2)

    bin_midpoints_tensor = torch.tensor(binned["bin_midpoint"].values, dtype=torch.float32)
    weight_tensor = list(binned["weight_val"].values)
    #tensor_dict[event] = (bin_midpoints_tensor, weight_tensor)
    tensor_dict[event] = weight_tensor
    #print(np.shape(weight_tensor))
    cat = list(group['category'].unique())
    data_y.append(cat_dict[cat[0]])

data_x = list(tensor_dict.values())
# Simulated dataset
num_samples = len(data_y)
num_bins = 256  # Angular bins
freq = Counter(data_y)
print(freq)

# Generate random data
#X_data = np.random.rand(num_samples, num_bins).astype(np.float32)  # Energy profiles
#y_data = np.random.randint(0, num_classes, size=(num_samples,))  # Labels

# Convert to tensors
X_tensor = torch.tensor(data_x, dtype=torch.float32)
y_tensor = torch.tensor(data_y).reshape(-1,1)
# Dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
# Define models
mlp_model = MLPClassifier(input_dim=num_bins, num_classes=num_classes)
cnn_model = CNNClassifier(input_dim=num_bins, num_classes=num_classes)
transformer_model = TransformerClassifier(seq_len=num_bins, num_classes=num_classes)

# Training function
def train_model(model, train_loader, num_epochs=200, lr=4.e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        ### ---- TRAINING PHASE ---- ###
        model.train()  # Set model to training mode
        total_train_loss = 0
        correct_train, total_train = 0, 0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze(1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += (predicted == y_batch.flatten()).sum().item()
            total_train += y_batch.size(0)
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        ### ---- VALIDATION PHASE ---- ###
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_val, total_val = 0, 0

        with torch.no_grad():  # Disable gradient computation for validation
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward pass
                logits = model(X_batch)
                loss = criterion(logits, y_batch.squeeze(1))

                # Compute accuracy
                _, predicted = torch.max(logits, dim=1)
                correct_val += (predicted == y_batch.flatten()).sum().item()
                total_val += y_batch.size(0)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Store losses & accuracies
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Accuracy:   {val_accuracy:.4f}") 

## Train all models
print("Training MLP:")
train_model(mlp_model, train_loader)

print("Training CNN:")
train_model(cnn_model, train_loader)

print("Training Transformer:")
train_model(transformer_model, train_loader)

