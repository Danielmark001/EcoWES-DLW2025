from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm():
    # Define the LSTM model
    model = Sequential()   

    # LSTM layers
    model.add(LSTM(units=64, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer (predict the next time step)
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
