import numpy as np
from pathlib import Path
from statistics import mean
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

dataset_path = Path("/home/dataset/project2")

LOGISITC_REPEAT_DICT = {
    "train_size": [0.2, 0.4, 0.6, 0.8],
    "vectorizer": [CountVectorizer(), TfidfVectorizer()],
    "solver": ["newton-cg", "sag", "saga", "lbfgs"],
    "max_iteration": [1000, 3000, 9000, 12000],
    "tol": [1, 0.01, 0.001],
}


NAIVE_BAYES_REPEAT_DICT = {
        "vectorizer": [CountVectorizer(), TfidfVectorizer()]
}


def evaluate_acc(test, pred):
    return np.sum(pred == test) / test.shape[0]


def print_acc_err(res):
    try:
        acc = res[0]
        err = res[1]
        logging.info("ACC: avg: {}, {}".format(mean(acc), acc))
        logging.info("ERR: avg: {}, {}".format(mean(err), err))
    except:
        pass