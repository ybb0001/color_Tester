import tensorflow as tf


class Utils:
    @staticmethod
    def find_last_conv_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv Layer in model!")
