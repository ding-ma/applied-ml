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
        model_config,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
    ):
        super().__init__(model_config["input_dim"], model_config["output_dim"], model_config["output_fnc"],learn_rate_init, batch_size, reg_lambda, anneal)
        
        self.hidden_1_dim: int = model_config["hidden_1_dim"]
        self.hidden_2_dim: int = model_config["hidden_2_dim"]
        self.hiddent_1_fnc: ActivationFunction = model_config["hiddent_1_fnc"]
        self.hiddent_2_fnc: ActivationFunction = model_config["hiddent_2_fnc"]
        
        self.W1 = np.random.randn(self.input_dim, self.hidden_1_dim) / np.sqrt(self.input_dim)
        self.b1 = np.zeros((1, self.hidden_1_dim))
        
        self.W2 = np.random.randn(self.hidden_1_dim, self.hidden_2_dim) / np.sqrt(self.hidden_1_dim) 
        self.b2 = np.zeros((1, self.hidden_2_dim)) 
        
        self.W3 = np.random.randn(self.hidden_2_dim, self.output_dim) / np.sqrt(self.hidden_2_dim) 
        self.b3 = np.zeros((1, self.output_dim)) 
 

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
                else:  # return to beginning
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
                
                # Forward prop
                z1 = X.dot(self.W1) + self.b1
                a1 = np.tanh(z1)  # currently using tanh activation
                z2 = a1.dot(self.W2) + self.b2
                a2 = np.tanh(z2)
                z3 = a2.dot(self.W3) + self.b3

                delta = self.output_fnc(z3)
                # Backprop
                delta[range(num_data), y] -= 1

                dW3 = (a2.T).dot(delta)
                db3 = np.sum(delta, axis=0, keepdims=True)
                delta = delta.dot(self.W3.T) * (1 - np.power(a2, 2))  # tanh gradient

                dW2 = (a1.T).dot(delta)
                db2 = np.sum(delta, axis=0, keepdims=True)
                delta = delta.dot(self.W2.T) * (1 - np.power(a1, 2))  # tanh gradient

                dW1 = np.dot(X.T, delta)
                db1 = np.sum(delta, axis=0)

                # Add regularization to weights
                dW1 += self.reg_lambda * self.W1
                dW2 += self.reg_lambda * self.W2
                dW3 += self.reg_lambda * self.W3

                # Gradient descent updates
                self.W1 += -learn_rate * dW1
                self.b1 += -learn_rate * db1
                self.W2 += -learn_rate * dW2
                self.b2 += -learn_rate * db2
                self.W3 += -learn_rate * dW3
                self.b3 += -learn_rate * db3

                if i % 500 == 0:
                    loss = self.compute_ce_loss(X, y)
                    running_loss += loss
                    logging.info("Loss at epoch %d iteration %d: %f" % (epoch, i, loss))

            loss_item = running_loss / int(self.num_iterations / 2000)
            logging.info("Loss at epoch %d: %f" % (epoch, loss_item))
            self.loss_history.append((epoch, loss_item))

            train_acc = self.compute_acc(train_array, train_labels_array)
            logging.info("Train_acc at epoch %d: %f" % (epoch, train_acc))
            self.train_acc_history.append((epoch, train_acc))

            test_acc = self.compute_acc(x_test, y_test)
            logging.info("Test_acc at epoch %d: %f" % (epoch, test_acc))
            self.test_acc_history.append((epoch, test_acc))

    def compute_ce_loss(self, X,y):
        num_data = X.shape[0]

        z1 = X.dot(self.W1) + self.b1
        a1 = self.hiddent_1_fnc(z1)  # currently using tanh activation

        z2 = a1.dot(self.W2) + self.b2
        a2 = self.hiddent_2_fnc(z2)
    
        z3 = a2.dot(self.W3) + self.b3
        initial_probs = self.output_fnc(z3)

        # Compute cross-entropy loss
        log_probs = -np.log(initial_probs[range(num_data), y])
        loss = np.sum(log_probs)

        # Add regularization
        loss += (self.reg_lambda / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))

        out = (1.0 / num_data) * loss
        return out    

    def predict(self, x):
    
        z1 = x.dot(self.W1) + self.b1
        a1 = self.hiddent_1_fnc(z1)

        z2 = a1.dot(self.W2) + self.b2
        a2 = self.hiddent_2_fnc(z2)

        z3 = a2.dot(self.W3) + self.b3
        probs = self.output_fnc(z3)

        labels = np.argmax(probs, axis=1) 
        return labels