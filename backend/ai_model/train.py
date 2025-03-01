import pandas as pd
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(data_path):
    """Train the model using preprocessed data."""
    # Load preprocessed data
    data = pd.read_csv(data_path)
    
    # Split into features (X) and target (y)
    X = data[['NOx', 'PM', 'SOx', 'Energy']]  # Select features
    y = data['CO2']  # Target variable
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(input_shape=X_train.shape[1])
    
    # Set up checkpoint to save the best model
    checkpoint = ModelCheckpoint('saved_models/energy_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])

    print("Model training complete. Best model saved to 'saved_models/energy_model.h5'.")

if __name__ == "__main__":
    train_model('preprocessed_data.csv')
