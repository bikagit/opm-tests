
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MinMaxScaler(Layer):
    def __init__(self, **kwargs):
        super(MinMaxScaler, self).__init__(**kwargs)

    def call(self, inputs, min_vals, max_vals, mode='scale'):
        if mode == 'scale':
            scaled = (inputs - min_vals) / (max_vals - min_vals)
            return scaled
        elif mode == 'unscale':
            unscaled = inputs * (max_vals - min_vals) + min_vals
            return unscaled
        else:
            raise ValueError("Mode should be either 'scale' or 'unscale'")

    def get_config(self):
        config = super(MinMaxScaler, self).get_config()
        return config

# Usage Example

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Example data
data = np.random.rand(100, 10)
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)

# Define the model
inputs = Input(shape=(10,))
min_vals_input = Input(shape=(10,))
max_vals_input = Input(shape=(10,))

scaler = MinMaxScaler()
scaled = scaler(inputs, min_vals_input, max_vals_input, mode='scale')
# Add other layers here

model = Model(inputs=[inputs, min_vals_input, max_vals_input], outputs=scaled)

# Compile and fit the model
model.compile(optimizer='adam', loss='mse')
model.fit([data, min_vals, max_vals], data, epochs=10)

# Example of unscaling
rescaled_data = model.predict([data, min_vals, max_vals])
unscaled_data = scaler(rescaled_data, min_vals, max_vals, mode='unscale')
