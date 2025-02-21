<<<<<<< HEAD
=======
import torch
import torch.nn as nn

>>>>>>> 3a08454 (fixed)
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear((input_dim // 4) * 64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)  # Flatten for FC layer
        return self.fc(x)

