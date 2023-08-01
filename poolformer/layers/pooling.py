import tensorflow as tf 
import numpy as np 


class Pooling(tf.keras.layers.Layer):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    """

    def __init__(self, pool_size=3, **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size, strides=1, padding="same")

    def call(self, x):
        y = self.pool(x)
        return y - x