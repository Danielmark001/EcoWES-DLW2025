import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ====================================================
# 1. Data Loading and Preprocessing
# ====================================================

def load_and_preprocess(csv_path, sensor_columns, label_column):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    df = df[df["machine_status"].isin(["NORMAL", "RECOVERING"])]
    df["machine_status"] = df["machine_status"].replace({"NORMAL": 0, "RECOVERING": 1})
    
    # If timestamp exists, sort by it (and then drop it)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop(columns=["timestamp"])
    
    # Fill missing values (forward then backward)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Scale sensor columns
    scaler = StandardScaler()
    df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
    
    # Convert dataframe to numpy arrays
    data_array = df[sensor_columns].values    # shape: (num_samples, num_features)
    labels_array = df[label_column].values      # shape: (num_samples,)
    
    return data_array, labels_array, scaler

# ==========================
# 2. Dataset: Sliding Window
# ==========================
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
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output length after conv/pooling:
        conv1_out = window_size - 5 + 1      # after conv1
        pool1_out = conv1_out // 2           # after pool1
        conv2_out = pool1_out - 3 + 1        # after conv2
        pool2_out = conv2_out // 2           # after pool2
        
        # LSTM: input size = number of channels (64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, window_size, num_features)
        x = x.permute(0, 2, 1)   # now (batch, num_features, window_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1)   # now (batch, sequence_length, channels)
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        out = self.sigmoid(out)
        # Clamp the output strictly between 1e-7 and 1-1e-7
        out = torch.clamp(out, 1e-7, 1 - 1e-7)
        return out

# ====================================================
# 4. Training & Evaluation Functions
# ====================================================
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Debug: check output range
        if torch.min(outputs) < 0 or torch.max(outputs) > 1:
            print("Debug: Output range =", torch.min(outputs).item(), torch.max(outputs).item())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader.dataset), correct / total


# ====================================================
# 4. Main Training Pipeline
# ====================================================

if __name__ == '__main__':
    # Specify the path to the dataset in the subfolder "data"
    csv_path = os.path.join("datasets", "sensor.csv")
    
    # Hyperparameters
    window_size = 100       # Number of consecutive readings per sample
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Define sensor and label columns (adjust these as needed)
    sensor_columns = [f"sensor_0{i}" for i in range(10)]
    sensor_columns.extend([f"sensor_{i}" for i in range(10, 52)])
    label_column = "machine_status"   # Assumed binary label (after processing: e.g., 0 for normal, 1 for fault)
    
    # Load and preprocess the data
    data_array, labels_array, scaler = load_and_preprocess(csv_path, sensor_columns, label_column)
    print("Data shape:", data_array.shape, "Labels shape:", labels_array.shape)
    
    # Create sliding window dataset
    dataset = SlidingWindowDataset(data_array, labels_array, window_size)
    
    # Split the dataset into training and testing sets (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the CNN+LSTM model
    num_features = len(sensor_columns)
    model = CNNLSTM(num_features=num_features, window_size=window_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save the trained model for production
    model_save_path = "cnn_lstm_sensor_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")