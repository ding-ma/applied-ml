from datetime import datetime
from pathlib import Path
import numpy as np
import logging
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
from utils.activation import ActivationFunction, RUN_DATE
import logging
from model.AbstractMLP import AbstractMLP


class NoLayer(AbstractMLP):
    def __init__(
        self,
        model_config,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
        num_epochs=50,
        L2=False,
        early_stop=0,
    ):
        super().__init__(
            model_config["input_dim"],
            model_config["output_dim"],
            model_config["output_fnc"],
            learn_rate_init,
            batch_size,
            reg_lambda,
            anneal,
            num_epochs,
            L2,
            early_stop,
        )

        self.W1 = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b1 = np.zeros((1, self.output_dim))

        self.file_name = f"{RUN_DATE}_no_layer_L2({self.L2})_LR({self.learn_rate_init})_BS({self.batch_size})"

    def compute_loss(self, X, y):
        num_data = X.shape[0]
        z1 = X.dot(self.W1) + self.b1

        initial_probs = self.output_fnc(z1)
        loss = np.sum(-np.log(initial_probs[range(num_data), y]))

        if self.L2:
            loss += (self.reg_lambda / 2) * (np.sum(np.square(self.W1)))

        out = (1.0 / num_data) * loss
        return out

    def forward_prop(self, X):
        self.z1 = X.dot(self.W1) + self.b1

        self.delta = self.output_fnc(self.z1)

    def backward_pop(self, num_data, X, y, learn_rate):
        self.delta[range(num_data), y] -= 1

        dW1 = np.dot(X.T, self.delta)
        db1 = np.sum(self.delta, axis=0)

        dW1 += self.reg_lambda * self.W1
        self.W1 += -learn_rate * dW1
        self.b1 += -learn_rate * db1
