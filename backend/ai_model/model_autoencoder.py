from tensorflow.keras import layers, models

# Define the autoencoder model
def create_autoencoder():
    model = models.Sequential()
    # Encoder
    model.add(layers.Dense(32, activation='relu', input_shape=(1,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))

    # Decoder
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model
