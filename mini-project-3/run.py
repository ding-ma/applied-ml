from model.OneLayer import OneLayer
from utils.activation import *

import sys
from pathlib import Path
import tensorflow as tf

# import numpy as np
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


# make sure this dictionary has the same variable names as the constructor of MLP


data_preprocess_params = {
    "threshold": True,
    "normalize": True,
    "augment_data": {"rotate": True, "shift": True, "zoom": True, "shear": True, "all": True},
}

if __name__ == "__main__":
    np.random.seed(0)

    # turn lambda into string
    # params_to_log = MLP_params
    # params_to_log["activation_fnc"] = ":".join(
    #     inspect.getsourcelines(MLP_params["activation_fnc"])[0][0].strip().split(":")[1:]
    # )

    experiment_description = f"""
    some description about your experiment

    Experiment Parameters
    

    Preprocess Parameters
    {data_preprocess_params}
    """

    file_name = ""  # optional

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # TODO: uncooment when ready to run real tests
            # logging.FileHandler(filename=f"logs/{RUN_DATE}-{file_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # logging.info(experiment_description)

    # TODO: add kFold CV
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_array = x_train.reshape(x_train.shape[0], 28 * 28)  # 60,000 X 28 x 28 x 1
    test_array = x_test.reshape(x_test.shape[0], 28 * 28)


#     mlp = MLP(batch_size=10,base_learn_rate=0.008, reg_lambda=0.7)
    mlp = OneLayer(batch_size=20,learn_rate_init=0.0008, reg_lambda=0.7)
    mlp.init_model(1, {
        "input_dim": 28*28,
        "hidden_dim": 256,
        "output_dim": 10,
        "hiddent_fnc": TanH(),
        "output_fnc": Softmax()
    })

    mlp.fit(train_array, y_train, test_array, y_test, num_epochs=10)

    y_pred = mlp.predict(test_array)
    logging.info(f"Final test accuracy {mlp.compute_acc(test_array, y_test)}")
