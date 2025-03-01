#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
