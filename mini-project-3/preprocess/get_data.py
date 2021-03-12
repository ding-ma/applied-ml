# %%
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def aquire_data(threshold=True, remove_constant_px=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize dataset
    # source: https://www.tensorflow.org/quantum/tutorials/mnist
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
   
#   make image black or white. 
# TODO: arbitrary 0.5 value
    if threshold:
        black_or_white = np.vectorize(lambda x : 0 if x < 0.5 else 1)
        x_train = black_or_white(x_train)
        x_test = black_or_white(x_test)
        # x_test = black_or_white(x_test)
            

    return x_train, x_test, y_train, y_test


# %%
if __name__ == "__main__":
    aquire_data()
