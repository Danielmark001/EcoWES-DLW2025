import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

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
    
pd = pd.read_csv('datasets\\2025.csv')
pd = pd[pd["Gas Guzzler Exempt Desc (Where Truck = 1975 NHTSA truck definition)"] == 'Truck']
average_fuel = pd["Comb Unrd Adj FE - Conventional Fuel"].mean() # miles per gallon
average_fuel = average_fuel * 0.425144 # kilometers per liter
consumption_per_100m = 1 / (average_fuel * 0.1) # 0.1 per window
max_fuel = 80.0  # liters

def create_simulated_data(consumption_per_100m=consumption_per_100m, fuel_remaining=max_fuel):
    # Initialize simulation variables
    fuel_remaining = max_fuel
    distance = 0  # in meters
    data = []

    # Simulate fuel consumption every 100 m until the tank is empty
    while fuel_remaining > 0:
        # Add noise as a percentage variation (5% standard deviation)
        noise_factor = np.random.normal(loc=0, scale=0.05)
        # Compute actual consumption for this 100 m step
        consumption = consumption_per_100m * (1 + noise_factor)
        # Ensure consumption is not negative and does not exceed the fuel left
        consumption = max(consumption, 0)
        consumption = min(consumption, fuel_remaining)
        
        # Update fuel and distance
        fuel_remaining -= consumption
        distance += 100
        
        # Record this step (you can also record the consumption for inspection)
        data.append(consumption)

    # Create a DataFrame from the simulation
    return data

def get_simulated_data(data_points=500000, window_size=52):
    big_data = []
    for _ in range(65000): # Approximately 500k data points
        big_data += create_simulated_data()
    dataset = FuelDataset(big_data, window_size)
    return dataset