import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_simulation import get_simulated_data
from model import CNNAutoencoder, train_autoencoder

def main():
    dataset = get_simulated_data()
    batch_size = 256
    window_size = 52
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

if __name__ == "__main__":
    main()