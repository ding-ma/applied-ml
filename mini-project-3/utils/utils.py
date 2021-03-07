import numpy as np

def evaluate_acc(test, pred):
    return np.sum(pred == test) / test.shape[0]