import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from itertools import chain
import logging

def __apply_generator(img_gen, X_train, y_train):
    img_gen.fit(X_train, seed=1)
    return img_gen.flow(X_train, y_train, batch_size=1, shuffle=True)


def __rotate_image(X_train, y_train):
    img_gen = ImageDataGenerator(rotation_range=90, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __shift_image(X_train, y_train):
    img_gen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __zoom_image(X_train, y_train):
    img_gen = ImageDataGenerator(zoom_range=4, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __shear_image(X_train, y_train):
    img_gen = ImageDataGenerator(shear_range=150, fill_mode="nearest")
    return __apply_generator(img_gen, X_train, y_train)


def __apply_all_augmentation(X_train, y_train):
    img_gen = ImageDataGenerator(
        shear_range=5,
        zoom_range=0.5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=30,
        fill_mode="nearest",
    )
    return __apply_generator(img_gen, X_train, y_train)

def __gen_to_numpy(gen):
    logging.info("Generator to Numpy Start")
    x_list = []
    y_list = []
    for e in gen:
        x_list.append(e[0].reshape(1, 28*28))
        y_list.append(e[1])
    logging.info("Generator to Numpy End")
    return np.concatenate(x_list), np.array(y_list)

def aquire_data(threshold, normalize, augment_data):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    if normalize:
        # convert from int to float and normalize
        X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255

    # augment data!
    train_generator = chain(zip(X_train, y_train))
    # train_generator = chain()
    if augment_data.get("rotate", None):
        logging.info("Performing rotate")
        train_generator = chain(train_generator, __rotate_image(X_train, y_train))

    if augment_data.get("shift", None):
        logging.info("Performing shift")
        train_generator = chain(train_generator, __shift_image(X_train, y_train))

    if augment_data.get("zoom", None):
        logging.info("Performing zoom")
        train_generator = chain(train_generator, __zoom_image(X_train, y_train))

    if augment_data.get("shear", None):
        logging.info("Performing shear")
        train_generator = chain(train_generator, __shear_image(X_train, y_train))

    if augment_data.get("all", None):
        logging.info("Performing all")
        train_generator = chain(train_generator, __apply_all_augmentation(X_train, y_train))

    if threshold:
        logging.info("Performing threshold")
        black_or_white = np.vectorize(lambda x: 0 if x < 0.5 else 1)
        return __gen_to_numpy(map(lambda x: (black_or_white(x[0]), x[1]), train_generator)), X_test.reshape(X_test.shape[0], 28 * 28), y_test

    return __gen_to_numpy(train_generator), X_test.reshape(X_test.shape[0], 28 * 28), y_test
