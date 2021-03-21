# %%
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
from utils.activation import ActivationFunction
import logging
from model.AbstractMLP import AbstractMLP

class TwoLayer(AbstractMLP):
    def __init__(
        self,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
    ):
        super().__init__(learn_rate_init, batch_size, reg_lambda, anneal)
    
    def init_model(self, n_hidden_layers, model_config):
        pass

    def fit(self, train_array, train_labels_array, x_test=None, y_test=None, num_epochs=50):
        pass

    def compute_ce_loss(self, X,y):
        pass
    
    def predict(self, x):
        pass