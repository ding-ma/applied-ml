import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from backports.zoneinfo import ZoneInfo
from .Helpers import evaluate_acc

from statistics import mean
import logging


class CrossVal:
    def __init__(
        self,
        X: pd.Series,
        y: pd.Series,
        n_fold=5,
        loss_fnc=lambda y, yh: np.mean((y - yh) ** 2),
    ):
        self.X = X.rename("X")
        self.y = y.rename("y")
        self.n_fold = n_fold
        self.loss_fnc = loss_fnc

    def __len__(self):
        return (np.ceil(self.X.shape[0] / float(self.n_fold))).astype(int)

    def __cross_validation_split(self):
        for idx in range(self.n_fold):
            s = idx * len(self)
            e = (idx + 1) * len(self)
            logging.info("Starting CV {}/{} Test_Set[{}:{}]".format(idx, self.n_fold, s, e))

            #  recall that drop does not affect the original dataframe unless you put inplate=True
            x_train = self.X.drop(self.X.index[s:e])
            y_train = self.y.drop(self.y.index[s:e])

            x_test = self.X[s:e]
            y_test = self.y[s:e]
            yield x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    def kfoldCV_custom_size(self, model, vectorizer, train_size):
        """
        May not be able to perform the entire dataset CV.
        But it will perform K times with a random state
        There might be repeated datapoints in a train/test set.
        On average it should produce the same result
        """

        if not 0 < train_size < 1:
            raise ValueError("Train size needs to be within ]0,1[")

        combined = pd.concat([self.X, self.y], axis=1)
        kfold_acc = []
        kfold_err = []
        for fold in range(self.n_fold):
            logging.info(
                "Starting CV {}/{} Train={}, Test={}".format(
                    fold,
                    self.n_fold,
                    train_size,
                    (1 - train_size),
                )
            )

            test_set = combined.sample(frac=(1 - train_size))
            train_set = combined[~combined.isin(test_set)].dropna()

            x_train = train_set["X"].to_numpy()
            x_test = test_set["X"].to_numpy()

            y_train = train_set["y"].to_numpy()
            y_test = test_set["y"].to_numpy()

            model.fit(vectorizer.fit_transform(x_train), y_train)
            y_predict = model.predict(vectorizer.transform(x_test))
            acc = evaluate_acc(y_test, y_predict)
            err = self.loss_fnc(y_test, y_predict)
            kfold_acc.append(acc)
            kfold_err.append(err)
        return kfold_acc, kfold_err

    def kfoldCV(self, model, vectorizer, **kwargs):
        """
        model: NB, LR. your model needs to have fit and predict as functions at least
        vectorizer: CV, TFIDF
        """
        kfold_acc = []
        kfold_err = []

        for x_train, x_test, y_train, y_test in self.__cross_validation_split():
            # todo: might need to use batch trainer
            model.fit(vectorizer.fit_transform(x_train), y_train)
            y_predict = model.predict(vectorizer.transform(x_test))
            acc = evaluate_acc(y_test, y_predict)
            err = self.loss_fnc(y_test, y_predict)
            kfold_acc.append(acc)
            kfold_err.append(err)
        return kfold_acc, kfold_err

    def repeat(self, **kwargs):
        """
        *args, **kwargs

        pseudo: sklearn gridsearchcv

        self === this in java
        """
        # done in yyscratch
        pass