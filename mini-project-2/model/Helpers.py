import numpy as np
from pathlib import Path
from statistics import mean
import logging

dataset_path = Path("/home/dataset/project2")


def evaluate_acc(test, pred):
    return np.sum(pred == test) / test.shape[0]


def print_acc_err(res):
    acc = res[0]
    err = res[1]
    logging.info("ACC: avg: {}, {}".format(mean(acc), acc))
    logging.info("ERR: avg: {}, {}".format(mean(err), err))