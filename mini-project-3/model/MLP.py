# %%
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import logging


class Layer:
    def __init__(self, input_dim, output_dim, activation_fnc):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        # set them as none to make debugging easier
        self.q = None
        self.z = None
        self.a = None
        self.dW = None
        self.activation_fnc = activation_fnc

    def __str__(self):
        return f"W({self.W}), b({self.b}), q({self.q}), z({self.a}), dw({dW})"


class Softmax():
    def __call__(self, x):
        # e_x = np.exp(x)
        # Original code:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class MLP:
    def __init__(
        self,
        n_iterations=1000,
        base_learn_rate=0.01,
        output_activation=None,
        batch_size=10,
        reg_lambda=1e-2,
        anneal=True,
    ):
        self.n_iterations = n_iterations
        self.base_learn_rate = base_learn_rate
        self.batch_size = 10
        self.output_activation = output_activation
        self.reg_lambda = reg_lambda
        self.anneal = anneal

        self.layers = []
        self.model = {}
        self.loss_hist = []
        self.train_acc = []
        self.test_acc = []

    def fit(self, X, y, n_epochs):
        self.n_data = X.shape[0]
        n_iterations = int(self.n_data / self.batch_size)

        for epoch in range(n_epochs):
            curr_index = 0
            running_loss = 0

            for itr in range(n_iterations):

                # mini batch split
                x_slice = X[curr_index: curr_index + self.batch_size]
                y_slice = y[curr_index: curr_index + self.batch_size]

                # index calculation for batches
                if curr_index + 2 * self.batch_size <= self.n_data:
                    curr_index += self.batch_size
                else:
                    curr_index = 0

                if self.anneal:
                    if epoch < n_epochs:
                        learn_rate = self.base_learn_rate
                    elif epoch < n_epochs / 2:
                        learn_rate = self.base_learn_rate / 10
                    elif epoch < 4 * n_epochs / 5:
                        learn_rate = self.base_learn_rate / 100
                    else:
                        learn_rate = self.base_learn_rate / 1000
                else:
                    learn_rate = self.base_learn_rate

                # forward prob
                for layer in self.layers:
                    q = x_slice.dot(layer.W) + layer.b
                    z = layer.activation_fnc(q)
                    
                    layer.q = q
                    layer.z = z

                # z as out of scope it takes the last value
                # softmax output
                soft = Softmax()
                probs = soft(z)
                delta = probs
                delta -= 1
                # print(delta)

# delta[range(self.n_data), y_slice] -= 1
                # backprob and update weights
                for layer in reversed(self.layers):
                    # this might be reversed
                    dW = (layer.z.T).dot(delta)
                    db = np.sum(delta, axis=0, keepdims=True)
                    # TODO: bug need the previous a not the current one.
                    delta = delta.dot(layer.W.T) * \
                                      (1 - np.power(layer.z, 2))  # tanh gradient

                    layer.dW = self.reg_lambda * layer.W + dW
                    layer.W += -learn_rate * layer.dW
                    layer.b += -learn_rate * layer.db

                if not itr % 2000:
                    loss = self.__cross_entropy(x_slice, y_slice)
                    running_loss += loss
                    logging.info(
                        f"Loss at epoch {epoch}, iteration {itr}, loss {loss}")

            loss_items = running_loss/int(n_iterations/2000)
            logging.info(f"Loss at epoch {epoch}, {loss_items}")
            self.loss_hist.append((epoch, loss_items))

            train_acc = self.compute_train_acc()
            logging.info(f"Train accurary({train_acc}) at epoch {epoch}")
            self.train_acc.append((epoch, train_acc))

            test_acc = self.compute_test_acc()
            logging.info(f"Test accurary({test_acc}) at epoch {epoch}")
            self.test_acc.append((epoch, test_acc))

    def __compute_probs(self, X):
        for layer in self.layers:
            layer.a = np.tanh(layer.z)
            layer.z = layer.activation_fnc(z)
            layer.activation_fnc(layer.z)

        return np.exp(self.layers[-1].z) / np.sum(np.exp(self.layers[-1].z), axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.__compute_probs(X), axis=1)

    # Compute model label prediction accuracy on all 50,000 train images
    def evaluate_acc(test, pred):
        return np.sum(pred == test)/test.shape[0]

    def add_layer(self, layer: Layer):
        """
        Add layer starting from the input layer
        """
        self.layers.append(layer)

    def __cross_entropy(self, X, y):

        # power = np.exp(self.layers[-1].z)
        initial_probs = self.__compute_probs(X)

        # Compute cross-entropy loss
        loss = np.sum(-np.log(initial_probs[range(self.n_data), y]))

        # regularization
        loss += self.layer[0].W * (self.reg_lambda / 2)

        for layer in self.layers[1:]:
            loss += (np.sum(np.square(layer.W)))
        
        return (1 / self.n_data) * loss

