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
from .Helpers import evaluate_acc, print_acc_err
import multiprocessing

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
                "Starting 'CV' {}/{} Train={}, Test={}".format(
                    fold,
                    self.n_fold,
                    train_size,
                    (1 - train_size),
                )
            )

            test_set = combined.sample(frac=(1 - train_size))
            train_set = combined[~combined.isin(test_set)].dropna()

            x_train = train_set["X"]
            x_test = test_set["X"]

            y_train = train_set["y"].astype(int)
            y_test = test_set["y"].astype(int)

            model.fit(vectorizer.fit_transform(x_train), y_train)
            y_predict = model.predict(vectorizer.transform(x_test))
            acc = evaluate_acc(y_test, y_predict)
            err = self.loss_fnc(y_test, y_predict)
            kfold_acc.append(acc)
            kfold_err.append(err)
        return kfold_acc, kfold_err

    def kfoldCV(self, model, vectorizer):
        """
        model: NB, LR. your model needs to have fit and predict as functions at least (un-initialized!)
        vectorizer: CV, TFIDF
        """
        kfold_acc = []
        kfold_err = []

        for x_train, x_test, y_train, y_test in self.__cross_validation_split():
            # todo: might need to use batch trainer
            model.fit(vectorizer.fit_transform(x_train), y_train)
            y_predict = model.predict(vectorizer.transform(x_test))
            acc = evaluate_acc(y_test, y_predict)
            try:
                err = self.loss_fnc(y_test, y_predict)
            except:
                err = []
            kfold_acc.append(acc)
            kfold_err.append(err)
        return kfold_acc, kfold_err

    def repeat(self, model, parameters):
        """ Description: performs grid search on the given parameters

        :param model: Naive Bayes or Logistic regression (un-initialized!)
        :param parameters: Dictionary of various parameters as array
            NaiveBayes: 
            {
                train_size: [list],
                vectorizer: [CountVect, TFIDF]
            }
            LR: 
            {
                train_size: [list, between 0 and 1 excluded],
                vectorizer: [CountVect, TFIDF],
                solver: ["newton-cg", "sag", "saga"],
                max_iteration: [ints],
                tol: [list]
            }

        :rtype: best parameters for the model
        """
        if "train_size" in parameters:
            pass

        training = []

        if model.__name__ == "LogisticRegression":
            for max_itr in parameters['max_iteration']:
                for solver in parameters['solver']:
                    for vec in parameters['vectorizer']:
                        for tol in parameters['tol']:
                            run = "max_itr={}, solver={}, vect={}, tol={}".format(max_itr, solver, type(vec).__name__, tol)
                            logging.info(run)
                            res = self.kfoldCV(model(solver=solver, max_iter=max_itr, tol=tol), vec)
                            print_acc_err(res)
                            training.append((run, res))
                
        elif "NB" in model.__name__ or "Bayes" in model.__name__ :
            for vec in parameters['vectorizer']:
                run = "vect={}".format( type(vec).__name__)
                logging.info(run)
                res = self.kfoldCV(model(), vec)
                print_acc_err(res)
                training.append((run, res))
        else:
            raise ValueError("Can only be ran on Naive Bayes or Logistic Regression!")
            
        logging.info("Training complete!")
        logging.info(training)
        best = max(training,key=lambda x:x[1][0])
        logging.info(f"Best result is {best}")
        
        return best
    
    def repeat_custom_size(self, model, vect):
        """
        Use the best hyper parameters from repeat and we can test on various train size [0.2, 0.4, 0.6, 0.8]
        :param model with hyperparameters already initialized
        :vect CountVectorizer or Tfidf
        """
        
        training = []
        for train_size in [0.2, 0.4, 0.6, 0.8]:
            res = self.kfoldCV_custom_size(model, vect, train_size)
            print_acc_err(res)
            training.append((str(train_size), res))
        
        logging.info("Training complete!")
        logging.info(training)
        
        return training