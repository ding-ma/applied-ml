import pandas as pd
import numpy as np
import itertools
from sklearn.utils import shuffle
from datetime import datetime
from backports.zoneinfo import ZoneInfo
from sklearn.preprocessing import LabelEncoder
import logging


class BernoulliBayes:
    _alpha = 1.0
    _num_classes = 0
    _fit = None

    def __init__(self, alpha=1):
        self._alpha = alpha

    def fit(self, train_x, train_y):

        self._num_classes = np.amax(train_y) + 1

        # intialization of list containing count of occurrences of each class
        class_count = self._num_classes * [0]

        # count occurences of each class
        for i in train_y:
            class_count[i] = 1 + class_count[i]

        # initialization of matrix
        fit = np.zeros((self._num_classes, train_x.shape[1] + 1))

        # fills matrix with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._num_classes):
            for n, element in enumerate(train_y):
                if element == i:
                    fit[i, :-1] = train_x[n] + fit[i, :-1]
            likelihood = (fit[i, :-1] + self._alpha) / (float(class_count[i]) + 2.0 * self._alpha)
            fit[i, :-1] = likelihood
            prior = class_count[i] / train_x.shape[0]
            fit[i, -1] = prior

        self._fit = fit

    def predict(self, val_x, val_y):

        res = np.zeros((self._num_classes, val_x.shape[0]), dtype=np.float32)

        # adding class prior probability
        for C in range(self._num_classes):
            log_neg = 1 - self._fit[C, -1]
            prior = self._fit[C, -1]

            res[C] += np.log(prior / log_neg)

        likelihood = self._fit[:, :-1]
        res += np.log(likelihood) @ val_x.T
        res += (np.log(1 - likelihood).sum(axis=1).reshape((-1, 1))) - (np.log(1 - likelihood) @ val_x.T)

        return res.T

        # print(res.T)
        # print("what is this", self._fit[:, :-1])

        # predictions = []
        # for example in res.T:
        #     predictions.append(np.argmax(example))

        # return predictions
        # print("predictions", predictions)

        # print("accuracy: " + str(np.sum(predictions == val_y)/len(predictions)))


class MultiNomialBayes:
    _alpha = 1.0
    _num_classes = 0
    _fit = None

    def __init__(self, alpha=1):
        self._alpha = alpha

    def fit(self, train_x, train_y):

        # _num_classes is C in TA's code
        self._num_classes = np.amax(train_y) + 1

        # generates list containing a count of each class occurrence
        # occurences is Nc in TA's code
        class_count = self._num_classes * [0]

        for i in train_y:
            class_count[i] = 1 + class_count[i]

        fit = np.zeros((self._num_classes, train_x.shape[1] + 1))

        # print(class_count)

        # fills matrix with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._num_classes):
            for n, element in enumerate(train_y):
                if element == i:
                    fit[i, :-1] = train_x[n] + fit[i, :-1]

            # filling likelihoods for each entry
            likelihood = ((fit[i, :-1]) + self._alpha) / (float(class_count[i]) + train_x.shape[1] * self._alpha)
            fit[i, :-1] = likelihood
            # inserting prior in the last column of the array
            prior = class_count[i] / train_x.shape[0]
            fit[i, -1] = prior

        self._fit = fit

    def predict(self, val_x, val_y):
        # initializing matrix D*C
        res = np.zeros((self._num_classes, val_x.shape[0]), dtype=np.float32)

        for C in range(self._num_classes):
            prior = self._fit[C, -1]
            # prior_neg = 1 - prior
            prior = np.log(prior)
            res[C] += prior
        likelihood = self._fit[:, :-1]
        likelihood = np.log(likelihood) @ val_x.T
        res += likelihood

        return res.T

        # predictions = []
        # for example in res.T:
        #     predictions.append(np.argmax(example))

        # print("predictions", predictions)

        # print("accuracy: " + str(np.sum(predictions == val_y)/len(predictions)))
