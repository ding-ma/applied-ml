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
    """Process data.
    *Warning*, augmenting the dataset requires 16GB of ram
    :type threshold: bool
    :param threshold: Turn image blakc or white

    :type normalize: bool
    :param normalize: return normalized data

    :type augment_data:bool
    :param augment_data: augments train dataset by rotating and translating

    :raises:

    :rtype: np.array
    """  # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if normalize:
        # convert from int to float and normalize
        logging.info("Performing normalization")
        X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255

    if augment_data:
        logging.info("Performing data augmentation")

        # simple data augmentation
        r_left_10 = __rotate(-10, X_train)
        r_left_15 = __rotate(-15, X_train)
        r_right_10 = __rotate(10, X_train)
        r_right_15 = __rotate(15, X_train)
        s_down_1_1 = __shift((1, 1), X_train)
        s_up_1_1 = __shift((-1, -1), X_train)

        # 360k datapoints after
        X_train = np.concatenate([r_left_10, r_left_15, r_right_10, r_right_15, s_up_1_1, s_down_1_1])
        y_train = np.concatenate([y_train] * 6)

    if threshold:
        logging.info("Performing threshold")
        black_or_white = np.vectorize(lambda x: 0 if x < 0.5 else 1)
        return (
            black_or_white(X_train).reshape(X_train.shape[0], 28 * 28),
            y_train,
            X_test.reshape(X_test.shape[0], 28 * 28),
            y_test,
        )

    return X_train.reshape(X_train.shape[0], 28 * 28), y_train, X_test.reshape(X_test.shape[0], 28 * 28), y_test
