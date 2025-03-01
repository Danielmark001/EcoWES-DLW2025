#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================
# 1. Simulate Fuel Level Data
# ==========================
def simulate_fuel_data(num_points=2000, drop_probability=0.02, drop_amount=10.0):
    """
    Simulate fuel level data for a truck:
      - Starts at 100%.
      - Gradually decreases with a small constant consumption and random noise.
      - Occasionally injects a sudden drop to represent theft/leak.
    """
    data = []
    fuel = 100.0
    for i in range(num_points):
        # Normal consumption: subtract a small value plus Gaussian noise
        fuel -= 0.05 + np.random.normal(0, 0.01)
        # Occasionally inject a sudden drop
        if np.random.rand() < drop_probability:
            fuel -= drop_amount
        # Ensure fuel level is not negative
        fuel = max(fuel, 0)
        data.append(fuel)
    return np.array(data)

# ==========================
# 2. Create Sliding Windows Dataset
# ==========================
class FuelDataset(Dataset):
    """
    Creates sliding windows from the fuel level data.
    For an autoencoder, input equals target.
    """
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        # Reshape to (window_size, 1) as we have one channel (fuel level)
        return torch.tensor(window, dtype=torch.float32).unsqueeze(1)

# ==========================
# 3. Filter "Normal" Windows for Training
# ==========================
def filter_normal_windows(dataset, threshold=0.5):
    """
    Returns only windows where the difference (max-min) is below a threshold.
    """
    normal_windows = []
    for i in range(len(dataset)):
        window = dataset[i]
        if (window.max() - window.min()) < threshold:
            normal_windows.append(window)
    if len(normal_windows) > 0:
        return torch.stack(normal_windows)
    else:
        return None

# ==========================
# 4. Define the CNN Autoencoder
# ==========================
class CNNAutoencoder(nn.Module):
    def __init__(self, window_size):
        super(CNNAutoencoder, self).__init__()
        # Encoder: input shape (batch, 1, window_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # output: window_size/2
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # output: window_size/4
        )
        # Decoder: reconstruct the original window
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )
        
    def forward(self, x):
        # x: (batch, 1, window_size)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==========================
# 5. Training the Autoencoder
# ==========================
def train_autoencoder(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)  # shape: (batch, 1, window_size)
        batch = batch.permute(0, 2, 1) # shape: (batch, 1, window_size)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    return epoch_loss / len(dataloader.dataset)

# ==========================
# 6. Main Function
# ==========================
def main():
    # 1. Simulate fuel data
    fuel_data = simulate_fuel_data(num_points=2000, drop_probability=0.02, drop_amount=10.0)
    
    # Optionally plot the simulated fuel data
    plt.figure(figsize=(10,4))
    plt.plot(fuel_data, label="Fuel Level")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Fuel Level (%)")
    plt.title("Simulated Fuel Level Data")
    plt.legend()
    plt.show()
    
    # 2. Create the dataset with a sliding window
    window_size = 16  # e.g., 16 readings per window
    dataset = FuelDataset(fuel_data, window_size)
    print("Total windows:", len(dataset))
    
    # 3. Filter normal windows (assumes normal windows have little variation)
    normal_windows = filter_normal_windows(dataset, threshold=0.5)
    if normal_windows is None:
        print("No normal windows found. Adjust the threshold!")
        return
    print("Number of normal windows for training:", normal_windows.shape[0])
    
    # Create DataLoader for training
    batch_size = 32
    train_loader = DataLoader(normal_windows, batch_size=batch_size, shuffle=True)
    
    # 4. Define and instantiate the CNN Autoencoder
    model = CNNAutoencoder(window_size=window_size)
    
    # 5. Set up training parameters
    criterion = nn.MSELoss()  # Reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        loss = train_autoencoder(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Reconstruction Loss: {loss:.4f}")
    
    # 6. Save the trained model into a .pkl file
    model_save_path = "fuel_monitoring_model.pkl"
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
