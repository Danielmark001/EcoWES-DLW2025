import pandas as pd
from tensorflow.keras.models import load_model

def load_data(file_path):
    """Load new data to make predictions."""
    data = pd.read_csv(file_path)
    return data

def predict_emissions(data_path):
    """Make predictions using the trained model."""
    # Load the preprocessed data
    data = load_data(data_path)
    
    # Load the saved model
    model = load_model('saved_models/energy_model.h5')
    
    # Select features
    X = data[['NOx', 'PM', 'SOx', 'Energy']]
    
    # Make predictions
    predictions = model.predict(X)
    
    # Add predictions to the original data
    data['Predicted CO2'] = predictions
    print(data[['NOx', 'PM', 'SOx', 'Energy', 'Predicted CO2']])

if __name__ == "__main__":
    predict_emissions('preprocessed_data.csv')
