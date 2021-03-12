#%%
import joblib
from datetime import datetime
from pathlib import Path
from utils.utils import RUN_DATE


class MLP:
    def __init__(self, activation_fnc, weight, bias, n_hidden_layers=2, n_units_hidden_layers=64):
        self.activation_fnc = activation_fnc
        self.n_hidden_layers = n_hidden_layers
        self.n_units_hidden_layers = n_units_hidden_layers
        self.weight = weight
        self.bias = bias

    def fit(self, X_train, y_train, learning_rate, n_iter):
        # write your code

        ############ Pickle the model after it is fit #######################
        joblib.dump(self, Path("/home/dataset/project3").joinpath(f"{RUN_DATE}.pkl"), compress=1)

    def predict(self, X_test, y_test):
        pass


# %%
