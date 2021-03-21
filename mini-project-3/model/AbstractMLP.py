from abc import ABC, abstractmethod
from utils.activation import ActivationFunction, RUN_DATE
import joblib
from pathlib import Path
import logging

class AbstractMLP(ABC):
    def __init__(
        self,
        input_dim,
        output_dim,
        output_fnc,
        learn_rate_init=0.01,
        batch_size=4,
        reg_lambda=1e-2,
        anneal=True,
        num_epochs=50,
        L2=False
    ):
        self.learn_rate_init = learn_rate_init
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.anneal = anneal
        self.num_epochs = num_epochs
        self.L2 = L2
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_fnc: ActivationFunction = output_fnc
        
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.file_name = ""
    
    def fit(self, train_array, train_labels_array, x_test=None, y_test=None ):
        self.num_iterations = int(train_array.shape[0] / self.batch_size)
        for epoch in range(self.num_epochs):
            current_index = 0
            running_loss = 0
            # TODO: shuffle data
            for i in range(self.num_iterations):
                X = train_array[current_index : current_index + self.batch_size]
                y = train_labels_array[current_index : current_index + self.batch_size]

                if current_index + 2 * self.batch_size <= train_array.shape[0]:
                    current_index += self.batch_size
                else:  # return to beginning
                    current_index = 0

                num_data = X.shape[0]

                # Annealing schedule
                if epoch < self.num_epochs / 5:
                    learn_rate = self.learn_rate_init
                elif epoch < self.num_epochs / 2:
                    learn_rate = self.learn_rate_init / 10
                elif epoch < 4 * self.num_epochs / 5:
                    learn_rate = self.learn_rate_init / 100
                else:
                    learn_rate = self.learn_rate_init / 1000
                
                self.forward_prop(X)
                self.backward_pop(num_data, X, y, learn_rate)
                
                if i % 500 == 0:
                    loss = self.compute_loss(X, y)
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
            

    @abstractmethod
    def compute_loss(self, X):
        pass
    
    @abstractmethod
    def forward_prop(self, X,y):
        pass
    
    @abstractmethod
    def backward_pop(self, num_data, X, y, learn_rate):
        pass

    def compute_acc(self, X, y):
        correct = 0
        for i in range(X.shape[0]):
            predicted = self.predict(X[i])
            
            if predicted == y[i]:
                correct += 1
        
        accuracy = 100 * correct / X.shape[0]
        return accuracy

    @abstractmethod
    def predict(self, x):
        pass

    def save(self):
        save_path = Path().cwd().joinpath("pickles").joinpath(self.file_name + ".pkl")
        joblib.dump(self, save_path, compress=1)
