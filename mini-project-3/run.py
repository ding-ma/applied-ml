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


# data_preprocess_config = {
#     "threshold": False,
#     "normalize": True,
#     "augment_data": {"rotate": False, "shift": False, "zoom": False, "shear": False, "all": False}
# }

# gradient_config = {"batch_size": 10, "learn_rate_init": 0.0002, "reg_lambda": 0.1, "num_epochs": 15, "L2": False}

# model_config_0_layer = {
#     "input_dim": 28*28,
#     "output_dim": 10,
#     "output_fnc": Softmax(),
# }
# mlp = NoLayer(model_config_0_layer, **gradient_config)

# model_config_1_layer = {
#     "input_dim": 28 * 28,
#     "hidden_dim": 128,
#     "output_dim": 10,
#     "hiddent_fnc": ReLU(),
#     "output_fnc": Softmax(),
# }
# mlp = OneLayer(model_config_1_layer, **gradient_config)

if __name__ == "__main__":
    np.random.seed(0)
    
    experiments = [
        (
            {
                "input_dim": 28 * 28,
                "hidden_dim": 128,
                "output_dim": 10,
                "hiddent_fnc": ReLU(),
                "output_fnc": Softmax(),
            },
            {
                "batch_size": 10, "learn_rate_init": 0.0002, "reg_lambda": 0.1, "num_epochs": 15,  "L2": False, "anneal":True
            },
            {
                "threshold": False,
                "normalize": True,
                "augment_data": {"rotate": False, "shift": False, "zoom": False, "shear": False, "all": False}
            }
        ),
        (
            {
                "input_dim": 28 * 28,
                "hidden_dim": 128,
                "output_dim": 10,
                "hiddent_fnc": TanH(),
                "output_fnc": Softmax(),
            },
            {
                "batch_size": 10, "learn_rate_init": 0.0002, "reg_lambda": 0.1, "num_epochs": 15,  "L2": False, "anneal":True
            },
            {
                "threshold": False,
                "normalize": True,
                "augment_data": {"rotate": False, "shift": False, "zoom": False, "shear": False, "all": False}
            }
        ),
        (
            {
                "input_dim": 28 * 28,
                "hidden_dim": 128,
                "output_dim": 10,
                "hiddent_fnc": ReLU(),
                "output_fnc": Softmax(),
            },
            {
                "batch_size": 10, "learn_rate_init": 0.0002, "reg_lambda": 0.1, "num_epochs": 15,  "L2": False, "anneal":True
            },
            {
                "threshold": False,
                "normalize": False,
                "augment_data": {"rotate": False, "shift": False, "zoom": False, "shear": False, "all": False}
            }
        ),
        (
            {
                "input_dim": 28 * 28,
                "hidden_dim": 128,
                "output_dim": 10,
                "hiddent_fnc": TanH(),
                "output_fnc": Softmax(),
            },
            {
                "batch_size": 10, "learn_rate_init": 0.0002, "reg_lambda": 0.1, "num_epochs": 15,  "L2": False, "anneal":True
            },
            {
                "threshold": False,
                "normalize": False,
                "augment_data": {"rotate": False, "shift": False, "zoom": False, "shear": False, "all": False}
            }
        ),
    ]

    for model_config, gradient_config, preprocess_param in experiments:
        experiment_description = f"""
        Gradient Parameters
        {gradient_config}

        Preprocess Parameters
        {preprocess_param}

        Model Parameters
        {model_config}
        """

        mlp = OneLayer(model_config, **gradient_config)

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


        (train_array, y_train), test_array, y_test = aquire_data(**preprocess_param)

        mlp.fit(train_array, y_train, test_array, y_test)
        y_pred = mlp.predict(test_array)

        logging.info(f"Final test accuracy {mlp.compute_acc(y_pred, y_test)}")
        mlp.save()
        mlp.plot(y_test, y_pred)

        print("############################################################################")
