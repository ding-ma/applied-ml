from model.MLP import MLP
from utils.utils import evaluate_acc, RUN_DATE
import logging
import sys
from datetime import datetime
import inspect
from preprocess.get_data import aquire_data


# make sure this dictionary has the same variable names as the constructor of MLP
MLP_params = {
    "activation_fnc": lambda x: x + 1,
    "weight": 20000,
    "bias": 0.001,
    "n_hidden_layers": 2,
    "n_units_hidden_layers": 64,
}

data_preprocess_params = {
    "threshold": True,
    "augment_data": {"rotate": True, "shift": True, "zoom": True, "shear": True, "all": True},
}


# turn lambda into string
params_to_log = MLP_params
params_to_log["activation_fnc"] = ":".join(
    inspect.getsourcelines(MLP_params["activation_fnc"])[0][0].strip().split(":")[1:]
)

experiment_description = f"""
some description about your experiment

Experiment Parameters
{params_to_log}

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


logging.info(experiment_description)

# TODO: add kFold CV
train, test = aquire_data(**data_preprocess_params)

# model = MLP(**MLP_params)
# model.fit(1, 2, 3, 4)
