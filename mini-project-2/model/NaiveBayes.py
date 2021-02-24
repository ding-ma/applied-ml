import pandas as pd
import numpy as np
import itertools
from sklearn.utils import shuffle
from datetime import datetime
from backports.zoneinfo import ZoneInfo
from sklearn.preprocessing import LabelEncoder
import logging


class BernoulliBayes:
    _smoothing = 1.0
    _nclasses = 0
    _fitParams = None
    _encoder = LabelEncoder()

    def __init__(self, smoothing=1.0):
        self._smoothing = smoothing

    def fit(self, trainingSet, trainingLabels):

        self._nclasses = np.amax(trainingLabels) + 1

        # generates list containing a count of each class occurrence
        occurrences = [0] * self._nclasses

        for element in trainingLabels:
            occurrences[element] += 1

        # fit parameter matrix with shape (nclasses, nfeatures + 1)
        params = np.zeros((self._nclasses, trainingSet.shape[1] + 1))

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for n, element in enumerate(trainingLabels):
                if element == i:
                    params[i, :-1] += trainingSet[n]
            params[i, :-1] = (params[i, :-1] + self._smoothing) / (float(occurrences[i]) + 2.0 * self._smoothing)
            params[i, -1] = occurrences[i] / trainingSet.shape[0]

        self._fitParams = params

    def predict(self, validationSet):

        # ###############predicit###############
        # creating a log odds matrix
        odds = np.zeros((self._nclasses, validationSet.shape[0]), dtype=np.float32)

        # adding class prior probability
        for Class in range(self._nclasses):
            log_neg = 1 - self._fitParams[Class, -1]
            odds[Class] += np.log(self._fitParams[Class, -1] / log_neg)

        odds += np.log(self._fitParams[:, :-1]) @ validationSet.T
        odds += (np.log(1 - self._fitParams[:, :-1]).sum(axis=1).reshape((-1, 1))) - (
            np.log(1 - self._fitParams[:, :-1]) @ validationSet.T
        )

        predictions = []
        for example in odds.T:
            predictions.append(np.argmax(example))

        return predictions


class MultiNomialBayes:
    _smoothing = 1.0
    _nclasses = 0
    _fitParams = None
    _encoder = LabelEncoder()

    def __init__(self, smoothing=1.0):
        self._smoothing = smoothing

    def fit(self, trainingSet, trainingLabels):

        # _nclasses is C in TA's code
        self._nclasses = np.amax(trainingLabels) + 1

        # generates list containing a count of each class occurrence
        # occurences is Nc in TA's code
        occurrences = [0] * self._nclasses

        for element in trainingLabels:
            occurrences[element] += 1

        params = np.zeros((self._nclasses, trainingSet.shape[1] + 1))

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for n, element in enumerate(trainingLabels):
                if element == i:
                    params[i, :-1] += trainingSet[n]
            # filling likelihoods for each entry
            params[i, :-1] = ((params[i, :-1]) + self._smoothing) / (
                float(occurrences[i]) + trainingSet.shape[1] * self._smoothing
            )
            # inserting prior in the last column of the array
            params[i, -1] = occurrences[i] / trainingSet.shape[0]

        self._fitParams = params

    def predict(self, validationSet):

        # creating a log odds matrix
        odds = np.zeros((self._nclasses, validationSet.shape[0]), dtype=np.float32)

        # adding class prior probability
        for Class in range(self._nclasses):
            prior = self._fitParams[Class, -1]
            prior_neg = 1 - prior
            prior = np.log(prior)
            odds[Class] += prior
        likelihood = self._fitParams[:, :-1]
        likelihood = np.log(likelihood) @ validationSet.T
        odds += likelihood

        predictions = []
        for example in odds.T:
            predictions.append(np.argmax(example))

        return predictions
