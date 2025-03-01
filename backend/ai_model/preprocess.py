import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load historical energy consumption data
def load_data(file_path):
    """
    Load dataset from a CSV file and return a DataFrame
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None


# Handle missing data
def handle_missing_data(data):
    """
    Handle missing data by filling with mean or removing rows with missing values
    """
    # Example: Fill missing values in 'energy_consumption' column with the mean
    data['energy_consumption'] = data['energy_consumption'].fillna(data['energy_consumption'].mean())
    data['carbon_emissions'] = data['carbon_emissions'].fillna(data['carbon_emissions'].mean())
    return data


# Feature scaling
def scale_data(data, columns_to_scale):
    """
    Normalize or scale specific columns using MinMaxScaler
    """
    scaler = MinMaxScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data


# Feature engineering
def add_features(data):
    """
    Add new features to the dataset, e.g., extracting hour or day from the timestamp
    """
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    return data


if __name__ == "__main__":
    # Load the datasets
    energy_data = load_data('data/historical_energy_data.csv')
    emissions_data = load_data('data/emissions_data.csv')

    # Ensure the data is loaded before processing
    if energy_data is not None and emissions_data is not None:
        # Preprocess the energy and emissions data
        energy_data = handle_missing_data(energy_data)
        emissions_data = handle_missing_data(emissions_data)

        # Merge the datasets if needed
        merged_data = pd.merge(energy_data, emissions_data, on='timestamp', how='inner')

        # Add features
        merged_data = add_features(merged_data)

        # Scale selected columns
        scaled_data = scale_data(merged_data, columns_to_scale=['energy_consumption', 'carbon_emissions'])

        # Save the preprocessed data for use in the model
        scaled_data.to_csv('data/preprocessed_energy_emissions.csv', index=False)
        print("Preprocessing complete, data saved to 'data/preprocessed_energy_emissions.csv'")
