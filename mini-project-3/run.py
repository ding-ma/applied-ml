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
import joblib

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

# np.random.seed(0)


model_config = {
    "input_dim": 28 * 28,
    # "hidden_dim": 128,
    "hidden_1_dim": 512,
    "hidden_2_dim": 256,
    "output_dim": 10,
    # "hiddent_fnc": ReLU(),
    "hiddent_1_fnc": ReLU(),
    "hiddent_2_fnc": ReLU(),
    "output_fnc": Softmax(),
}

gradient_config = {
    "batch_size": 50,
    "learn_rate_init": 0.002,
    "reg_lambda": 0.1,
    "num_epochs": 20,
    "L2": True,
    "anneal": True,
    "early_stop": 0,
}

preprocess_param = {
    "threshold": False,
    "normalize": True,
    "augment_data": True,
}

pickle:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-29_135101_two_layer_512_ReLU_256_ReLU_L2(True)_LR(0.002)_BS(50).pkl")
mlp = TwoLayer(model_config, **gradient_config)

mlp.W1 = pickle.W1
mlp.W2 = pickle.W2
mlp.W3 = pickle.W3

mlp.b1 = pickle.b1
mlp.b2 = pickle.b2
mlp.b3 = pickle.b3

mlp.file_name += "_augmented_dataset(360k)"

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f"logs/{mlp.file_name}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

experiment_description = f"""
Copied weights and bias from 03-29_135101_two_layer_512_ReLU_256_ReLU_L2(True)_LR(0.002)_BS(50).pkl
and going to train on augmented dataset

Gradient Parameters
{gradient_config}

Preprocess Parameters
{preprocess_param}

Model Parameters
{model_config}
"""
logging.info(experiment_description)

train_array, y_train, test_array, y_test = aquire_data(**preprocess_param)

mlp.fit(train_array, y_train, test_array, y_test)
mlp.save()

y_pred = mlp.predict(test_array)

logging.info(f"Final test accuracy {mlp.compute_acc(y_pred, y_test)}")
mlp.save()
mlp.plot(y_test, y_pred)
