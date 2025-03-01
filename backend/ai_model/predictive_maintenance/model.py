import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    """
    Creates sliding windows from the sensor data.
    Each sample is a window of consecutive sensor readings and the label of the last time step.
    """
    def __init__(self, data, labels, window_size):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        # Window of sensor readings
        X = self.data[idx : idx + self.window_size]
        # Label from the last time step in the window
        y = self.labels[idx + self.window_size - 1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class CNNLSTM(nn.Module):
    def __init__(self, num_features, window_size):
        super(CNNLSTM, self).__init__()
        # CNN layers (Conv1d expects input shape: [batch, channels, sequence_length])
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        conv1_out = window_size - 5 + 1
        pool1_out = conv1_out // 2
        conv2_out = pool1_out
        pool2_out = conv2_out // 2
        conv3_out = pool2_out - 3 + 1
        pool3_out = conv3_out // 2

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu(self.conv1(x)))
        print("pool 1 x shape:", x.shape)
        x = self.pool2(self.relu(self.conv2(x)))
        print("pool 2 x shape:", x.shape)
        x = self.pool3(self.relu(self.conv3(x)))
        print("pool 3 x shape:", x.shape)

        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.relu(self.fc1(h_n[-1]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out