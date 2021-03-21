# %%
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


class OneLayer(AbstractMLP):
    def __init__(
        self,
        model_config,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
        num_epochs=50,
        L2 = False,
    ):
        super().__init__(model_config["input_dim"], model_config["output_dim"], model_config["output_fnc"],learn_rate_init, batch_size, reg_lambda, anneal,num_epochs,L2)
        self.hidden_dim: int = model_config["hidden_dim"]
        self.hiddent_fnc: ActivationFunction = model_config["hiddent_fnc"]

        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))

        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))

        self.file_name =  f"{RUN_DATE}_one_layer_{self.hidden_dim}_{self.hiddent_fnc}"


    def forward_prop(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.hiddent_fnc(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2

        self.delta = self.output_fnc(self.z2)    

    def backward_pop(self, num_data, X, y, learn_rate):
        self.delta[range(num_data), y] -= 1

        dW2 = (self.a1.T).dot(self.delta)
        db2 = np.sum(self.delta, axis=0, keepdims=True)

        self.delta = self.delta.dot(self.W2.T) * self.hiddent_fnc.gradient(self.a1)
        dW1 = np.dot(X.T, self.delta)
        db1 = np.sum(self.delta, axis=0)

        dW1 += self.reg_lambda * self.W1
        dW2 += self.reg_lambda * self.W2

        # Gradient descent updates
        self.W1 += -learn_rate * dW1
        self.b1 += -learn_rate * db1
        self.W2 += -learn_rate * dW2
        self.b2 += -learn_rate * db2


    def compute_loss(self, X, y):
        num_data = X.shape[0]

        # Forward prop to compute predictions
        z1 = X.dot(self.W1) + self.b1
        a1 = self.hiddent_fnc(z1) 

        z2 = a1.dot(self.W2) + self.b2
        initial_probs = self.output_fnc(z2)

        # Compute cross-entropy loss
        loss = np.sum(-np.log(initial_probs[range(num_data), y]))

        # Add regularization
        if self.L2:
            loss += (self.reg_lambda / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))

        out = (1.0 / num_data) * loss
        return out

    def predict(self, x):
        self.forward_prop(x)
        return np.argmax(self.delta, axis=1)
    