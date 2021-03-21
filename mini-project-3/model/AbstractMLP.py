from abc import ABC, abstractmethod
from utils.activation import ActivationFunction

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
    ):
        self.learn_rate_init = learn_rate_init
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.anneal = anneal
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_fnc: ActivationFunction = output_fnc
        
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
    
    @abstractmethod
    def fit(self, train_array, train_labels_array, x_test=None, y_test=None, num_epochs=50):
        pass

    @abstractmethod
    def compute_ce_loss(self, X,y):
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
