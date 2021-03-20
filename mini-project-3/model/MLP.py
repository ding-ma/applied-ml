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


class Layer:
    def __init__(self, input_dim, output_dim, activation_fnc):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))

        # set them as none to make debugging easier
        # z = W*x+b
        self.z = None

        # a = activation(q)
        self.a = None
        self.dW = None
        self.activation_fnc = activation_fnc

    def __str__(self):
        return f"W({self.W.shape}), b({self.b.shape}), a({self.a}), dw({self.dW})"

    def __repr__(self):
        return f"W({self.W.shape}), b({self.b.shape}), a({self.a}), dw({self.dW})"


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax:
    def __call__(self, x):
        # e_x = np.exp(x)
        # Original code:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH:
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class MLP:
    def __init__(
        self,
        base_learn_rate=0.01,
        output_activation=None,
        batch_size=2,
        reg_lambda=1e-2,
        anneal=True,
    ):
        self.base_learn_rate = base_learn_rate
        self.batch_size = batch_size
        self.output_activation = output_activation
        self.reg_lambda = reg_lambda
        self.anneal = anneal

        self.layers = []
        self.model = {}
        self.loss_hist = []
        self.train_acc = []
        self.test_acc = []

    def fit(self, x_train, y_train, x_test, y_test, n_epochs=50):
        n_iterations = int(x_train.shape[0] / self.batch_size)

        for epoch in range(n_epochs):
            curr_index = 0
            running_loss = 0

            for itr in range(n_iterations):

                # mini batch split
                x_slice = x_train[curr_index : curr_index + self.batch_size]
                y_slice = y_train[curr_index : curr_index + self.batch_size]
                self.n_data = x_slice.shape[0]

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
                z = x_slice.dot(self.layers[0].W) + self.layers[0].b
                a = self.layers[0].activation_fnc(z)
                self.layers[0].a = a
                for layer in self.layers[1:]:
                    z = a.dot(layer.W) + layer.b
                    a = layer.activation_fnc(z)
                    layer.a = a

                probs = layer.activation_fnc(z)
                delta = probs
                delta[range(self.n_data), y_slice] -= 1

                # backprob and update weights
                for i in range(self.n_hidden_layers, 0, -1):
                    # print(i, self.layers[i-1].activation_fnc)
                    self.layers[i].dW = (self.layers[i - 1].a.T).dot(delta)
                    self.layers[i].db = np.sum(delta, axis=0, keepdims=True)
                    delta = delta.dot(self.layers[i].W.T) * self.layers[i].activation_fnc.gradient(
                        self.layers[i - 1].a
                    )
                    self.layers[i].dW += self.reg_lambda * self.layers[i].W
                    self.layers[i].W += -learn_rate * self.layers[i].dW
                    self.layers[i].b += -learn_rate * self.layers[i].db

                self.layers[0].dW = np.dot(x_slice.T, delta)
                self.layers[0].db = np.sum(delta, axis=0)
                self.layers[0].dW += self.reg_lambda * self.layers[0].W
                self.layers[0].W += -learn_rate * self.layers[0].dW
                self.layers[0].b += -learn_rate * self.layers[0].db

                if itr % 2000 == 0:
                    loss = self.__cross_entropy(x_slice, y_slice)
                    running_loss += loss
                    logging.info(f"Loss at epoch {epoch}, iteration {itr}, loss {loss}")


            loss_items = running_loss / int(n_iterations / 2000)
            logging.info(f"Loss at epoch {epoch}, {loss_items}")
            self.loss_hist.append((epoch, loss_items))

            train_acc = self.evaluate_acc(self.predict(x_train), y_train)
            logging.info(f"Train accurary({train_acc}) at epoch {epoch}")
            self.train_acc.append((epoch, train_acc))

            test_acc = self.evaluate_acc(self.predict(x_test), y_test)
            logging.info(f"Test accurary({test_acc}) at epoch {epoch}")
            self.test_acc.append((epoch, test_acc))

    def __compute_probs(self, X):

        for i in range(0, self.n_hidden_layers + 1):
            if i == 0:
                z = X.dot(self.layers[0].W) + self.layers[0].b
            else:
                W = self.layers[i].W
                b = self.layers[i].b
                z = self.layers[i - 1].a.dot(W) + b
            if i != self.n_hidden_layers:
                self.layers[i].a = self.layers[i].activation_fnc(z)
            else:
                z = self.layers[i].activation_fnc(z)

        return z

    def predict(self, X):
        return np.argmax(self.__compute_probs(X), axis=1)

    # Compute model label prediction accuracy on all 50,000 train images
    def evaluate_acc(self, test, pred):
        return np.sum(pred == test) / test.shape[0]

    def add_layer(self, layer: Layer):
        """
        Add layer starting from the input layer
        """
        self.layers.append(layer)
        self.n_hidden_layers = len(self.layers) - 1

    def __cross_entropy(self, X, y):

        # power = np.exp(self.layers[-1].z)
        initial_probs = self.__compute_probs(X)

        # Compute cross-entropy loss
        loss = np.sum(-np.log(initial_probs[range(self.n_data), y]))

        # regularization
        loss += (self.reg_lambda / 2) * (np.sum(np.square(self.layers[0].W)))

        for layer in self.layers[1:]:
            loss += np.sum(np.square(layer.W))

        out = (1 / self.n_data) * loss
        return out

    def __compute_acc(self, X, y):
        correct = 0
        total = 0
        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            if pred == y[i]:
                correct += 1
            total += 1
        return 100 * correct / total
