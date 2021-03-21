from model import *
from utils.activation import *

import sys
from pathlib import Path
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import logging
import sys
from datetime import datetime
import inspect

from preprocess.get_data import aquire_data
from itertools import islice
import numpy as np


data_preprocess_config = {
    "threshold": True,
    "normalize": True,
    "augment_data": {"rotate": True, "shift": True, "zoom": True, "shear": True, "all": True}
}

gradient_config = {"batch_size": 20, "learn_rate_init": 0.0008, "reg_lambda": 0.7}

if __name__ == "__main__":
    np.random.seed(0)

    experiment_description = f"""
    Augmented all the data

    Experiment Parameters
    {gradient_config}

    Preprocess Parameters
    {data_preprocess_config}
    """

    # model_config_0_layer = {
    #     "input_dim": 28*28,
    #     "output_dim": 10,
    #     "output_fnc": Softmax()
    # }
    # mlp = NoLayer(model_config_0_layer, **gradient_config)

    # model_config_1_layer = {
    #     "input_dim": 28 * 28,
    #     "hidden_dim": 256,
    #     "output_dim": 10,
    #     "hiddent_fnc": TanH(),
    #     "output_fnc": Softmax(),
    # }
    # mlp = OneLayer(model_config_1_layer, **gradient_config)
    model_config_2_layer = {
        "input_dim": 28 * 28,
        "hidden_1_dim": 128,
        "hidden_2_dim": 128,
        "output_dim": 10,
        "hiddent_1_fnc": ReLU(),
        "hiddent_2_fnc": ReLU(),
        "output_fnc": Softmax(),
    }

    mlp = TwoLayer(model_config_2_layer, **gradient_config)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(filename=f"logs/{mlp.file_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(experiment_description)


    (train_array, y_train), test_array, y_test = aquire_data(**data_preprocess_config)
    print(train_array.shape, y_train.shape, test_array.shape, y_test.shape)


    mlp.fit(train_array, y_train, test_array, y_test, num_epochs=40)
    y_pred = mlp.predict(test_array)
    logging.info(f"Final test accuracy {mlp.compute_acc(test_array, y_test)}")
    mlp.save()

