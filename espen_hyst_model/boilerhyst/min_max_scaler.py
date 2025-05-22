
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
