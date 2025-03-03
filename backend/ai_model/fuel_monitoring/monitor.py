import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

def detect_anomaly(data, model):
    """
    Detect anomalies in the fuel level data.
    """
    # Set the model to evaluation mode
    model.eval()
    # Compute the reconstruction error for each data point
    error = nn.MSELoss(model(data), data)
    if error < 0.005:
        return False
    else:  
        return True

def main():
    file_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "fuel_monitoring_model.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(file_path, map_location=device, pickle_module=pickle)
    # Load the data
    data = np.load("data.npy")
    # Detect anomalies
    anomalies = []
    for i in range(len(data)):
        if detect_anomaly(data[i], model):
            anomalies.append(i)
            print(f"Anomaly detected at index {i}")
    
if __name__ == "__main__":
    main()