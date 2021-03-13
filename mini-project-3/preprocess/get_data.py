import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from itertools import chain

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def __apply_generator(img_gen, X_train, y_train):
    np.random.seed(1)
    img_gen.fit(X_train, seed=1)
    x, y = unison_shuffled_copies(X_train, y_train)
    return img_gen.flow(x, y, batch_size=1)


def __rotate_image(X_train, y_train):
    np.random.seed(2)
    img_gen = ImageDataGenerator(rotation_range=0.3, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __shift_image(X_train, y_train):
    np.random.seed(3)
    img_gen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __zoom_image(X_train, y_train):
    np.random.seed(4)
    img_gen = ImageDataGenerator(zoom_range=0.3, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __shear_image(X_train, y_train):
    np.random.seed(5)
    img_gen = ImageDataGenerator(shear_range=0.03, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __apply_all_augmentation(X_train, y_train):
    np.random.seed(6)
    img_gen = ImageDataGenerator(
        shear_range=0.03,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.3,
        fill_mode="nearest",
    )
    return __apply_generator(img_gen, X_train, y_train)


def aquire_data(threshold, augment_data):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # convert from int to float and normalize
    X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255

    # make image black or white.
    if threshold:
        black_or_white = np.vectorize(lambda x: 0 if x < 0.5 else 1)
        X_train = black_or_white(X_train)
        X_test = black_or_white(X_test)

    # augment data!
    train_generator = chain()
    if augment_data.get("rotate", None):
        train_generator = chain(train_generator, __rotate_image(X_train, y_train))

    if augment_data.get("shift", None):
        train_generator = chain(train_generator, __shift_image(X_train, y_train))

    if augment_data.get("zoom", None):
        train_generator = chain(train_generator, __zoom_image(X_train, y_train))

    if augment_data.get("shear", None):
        train_generator = chain(train_generator, __shear_image(X_train, y_train))

    if augment_data.get("all", None):
        train_generator = chain(train_generator, __apply_all_augmentation(X_train, y_train))

    return train_generator, zip(X_test, y_test)
