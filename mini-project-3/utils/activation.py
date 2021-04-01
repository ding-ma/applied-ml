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
    def __call__(self, x, axis=1):
        # source: https://stackoverflow.com/questions/43401593/softmax-of-a-large-number-errors-out
        # save typing...
        kw = dict(axis=axis, keepdims=True)

        # make every value 0 or below, as exp(0) won't overflow
        xrel = x - x.max(**kw)

        # if you wanted better handling of small exponents, you could do something like this
        # to try and make the values as large as possible without overflowing, The 0.9
        # is a fudge factor to try and ignore rounding errors
        #
        xrel += np.log(np.finfo(float).max / x.shape[axis]) * 0.9

        exp_xrel = np.exp(xrel)
        return exp_xrel / exp_xrel.sum(**kw)

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
        return np.where(x > 0, x, 0)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)

    def __repr__(self):
        return "ReLU"
