import os
import numpy as np
import torch
import torch.nn as nn
import pickle

def predict_maintenance(data, model):
    model.eval()
    output = model(data)
    maintenance = []
    for out in output:
        if out > 0.5:
            maintenance.append(True)
        else:
            maintenance.append(False)
    return maintenance


def main():
    file_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "predictive_maintenance_model.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(file_path, map_location=device, pickle_module=pickle)
    # Load the data
    data = np.load("data.npy")
    # Predict maintenance
    maintenance = []
    for i in range(len(data)):
        if predict_maintenance(data[i], model):
            maintenance.append(i)
            print(f"Maintenance predicted at index {i}") 

if __name__ == "__main__":
    main()