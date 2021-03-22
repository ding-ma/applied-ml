#%%
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from itertools import chain
import logging
from scipy.ndimage import rotate, shift


def __rotate(degree, img):
    return np.array([rotate(i, degree, reshape=False) for i in img])

def __shift(pos, img):
    return np.array([shift(i, pos) for i in img])


def aquire_data(threshold, normalize, augment_data):
    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    if normalize:
        # convert from int to float and normalize
        logging.info("Performing normalization")
        X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255
    
    if augment_data:
        logging.info("Performing data augmentation")

        # simple data augmentation
        r_left = __rotate(-30, X_train)
        r_right = __rotate(30, X_train)
        s_down = __shift((2,2), X_train)
        s_up = __shift((-2,-2), X_train)

        # 300k datapoints after
        X_train = np.concatenate([r_left, r_right, s_up, s_down, X_train])
        y_train = np.concatenate([y_train]*5)

    if threshold:
        logging.info("Performing threshold")
        black_or_white = np.vectorize(lambda x: 0 if x < 0.5 else 1)
        return (
            black_or_white(X_train).reshape(X_train.shape[0], 28*28),
            y_train,
            X_test.reshape(X_test.shape[0], 28 * 28),
            y_test,
        )
    
    return X_train.reshape(X_train.shape[0], 28*28), y_train, X_test.reshape(X_test.shape[0], 28 * 28), y_test
