from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np

RUN_DATE = datetime.now().strftime("%m-%d_%H%M%S")

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

    def __repr__(self):
        return "Signmoid"

class Softmax(ActivationFunction):
    def __call__(self, x):
        # e_x = np.exp(x)
        power = np.exp(x)
        return power / np.sum(power, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
    
    def __repr__(self):
        return "Softmax"

class TanH(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.power(x, 2)
    
    def __repr__(self):
        return "TanH"

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)
    
    def __repr__(self):
        return "ReLU"