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
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def gradient(self, x):
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax(ActivationFunction):
    def __call__(self, x):
        # e_x = np.exp(x)
        power = np.exp(x)
        return power / np.sum(power, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.power(x, 2)


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class MLP:
    def __init__(
        self,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
    ):
        self.learn_rate_init = learn_rate_init
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.anneal = anneal

        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
    
    def init_model(self, n_hidden_layers, model_config):
        self.n_hidden_layers = n_hidden_layers
        
        if n_hidden_layers == 0:
            pass
        
        elif n_hidden_layers == 1:
            self.input_dim:int = model_config['input_dim']
            self.hidden_dim:int = model_config['hidden_dim']
            self.output_dim:int = model_config['output_dim']
            self.hiddent_fnc:ActivationFunction = model_config['hiddent_fnc']
            self.output_fnc:ActivationFunction = model_config['output_fnc']

            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
            self.b1 = np.zeros((1, self.hidden_dim))

            self.W2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)  
            self.b2 = np.zeros((1, self.output_dim))

        elif n_hidden_layers == 2:
            pass
        else:
            raise ValueError("Model depth can be 0, 1, or 2")
        

    def fit(self, train_array, train_labels_array, x_test=None, y_test=None, num_epochs=50):
        self.num_iterations = int(train_array.shape[0] / self.batch_size)
        
        for epoch in range(num_epochs):
            current_index = 0
            running_loss = 0
            for i in range(self.num_iterations):
                X = train_array[current_index : current_index + self.batch_size]
                y = train_labels_array[current_index : current_index + self.batch_size]
                
                if current_index + 2 * self.batch_size <= train_array.shape[0]:
                    current_index += self.batch_size
                else: # return to beginning
                     current_index = 0
                    
                num_data = X.shape[0]
                      # Annealing schedule
                if epoch < num_epochs / 5:
                    learn_rate = self.learn_rate_init
                elif epoch < num_epochs / 2:
                    learn_rate = self.learn_rate_init / 10
                elif epoch < 4 * num_epochs / 5:
                    learn_rate = self.learn_rate_init / 100
                else:
                    learn_rate = self.learn_rate_init / 1000
                
                z1 = X.dot(self.W1) + self.b1
                a1 = self.hiddent_fnc(z1)

                z2 = a1.dot(self.W2) + self.b2

                # Backprop
                delta = self.output_fnc(z2)
                delta[range(num_data), y] -= 1
                
                dW2 = (a1.T).dot(delta)
                db2 = np.sum(delta, axis=0, keepdims=True)
                
                delta = delta.dot(self.W2.T) * self.hiddent_fnc.gradient(a1)  # tanh gradient
                dW1 = np.dot(X.T, delta)
                db1 = np.sum(delta, axis=0)

                dW1 += self.reg_lambda * self.W1
                dW2 += self.reg_lambda * self.W2

                # Gradient descent updates
                self.W1 += -learn_rate * dW1
                self.b1 += -learn_rate * db1
                self.W2 += -learn_rate * dW2
                self.b2 += -learn_rate * db2


                if i % 2000 == 0: 
                    loss = self.compute_ce_loss(X, y)
                    running_loss += loss
                    print("Loss at epoch %d iteration %d: %f" % (epoch, i, loss))
        
            loss_item = running_loss / int(self.num_iterations / 2000) 
            print("Loss at epoch %d: %f" % (epoch, loss_item))   
            self.loss_history.append((epoch, loss_item))

            train_acc = self.compute_acc(train_array, train_labels_array)
            print("Train_acc at epoch %d: %f" % (epoch, train_acc))
            self.train_acc_history.append((epoch, train_acc))

            test_acc = self.compute_acc(x_test, y_test)
            print("Test_acc at epoch %d: %f" % (epoch, test_acc))
            self.test_acc_history.append((epoch, test_acc))

    def compute_ce_loss(self, X,y):
        num_data = X.shape[0]
        # Forward prop to compute predictions
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)  # currently using tanh activation
        # a1 = np.maximum(1e-3, z1)  # ReLU activation
        z2 = a1.dot(self.W2) + self.b2
        initial_probs = self.output_fnc(z2)

        # Compute cross-entropy loss
        loss = np.sum(-np.log(initial_probs[range(num_data), y]))
        
        # Add regularization
        loss += (self.reg_lambda / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))

        out = (1. / num_data) * loss
        return out
    
    def compute_acc(self, X, y):
        correct = 0
        for i in range(X.shape[0]):
            predicted = self.predict(X[i])
            
            if predicted == y[i]:
                correct += 1
        
        accuracy = 100 * correct / X.shape[0]
        return accuracy

    def predict(self, x):
            # Forward prop to compute predictions
        z1 = x.dot(self.W1) + self.b1
        a1 = np.tanh(z1)  # currently using tanh activation
        # a1 = np.maximum(1e-3, z1)  # leaky ReLU activation
        z2 = a1.dot(self.W2) + self.b2
        probs = self.output_fnc(z2)
        
        labels = np.argmax(probs, axis=1) 
        return labels