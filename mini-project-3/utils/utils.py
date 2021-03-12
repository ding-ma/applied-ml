import numpy as np
from datetime import datetime

def evaluate_acc(test, pred):
    return np.sum(pred == test) / test.shape[0]

RUN_DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")