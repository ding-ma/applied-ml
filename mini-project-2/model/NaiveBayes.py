import pandas as pd
import numpy as np
import itertools
from sklearn.utils import shuffle
from datetime import datetime
from backports.zoneinfo import ZoneInfo
import logging


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, x, y):
        logging.info("Sparse to Np.arr")
        x = x.todense()
        y = y.todense()
        logging.info("Starting Fit")
        # Unmodified teacher's code
        N, D = x.shape
        C = np.max(y) + 1
        # one parameter for each feature conditioned on each class
        mu, sigma = np.zeros((C, D)), np.zeros((C, D))
        Nc = np.zeros(C)  # number of instances in class c
        # for each class get the MLE for the mean and std
        for c in range(C):
            x_c = x[y == c]  # slice all the elements from class c
            Nc[c] = x_c.shape[0]  # get number of elements of class c
            mu[c, :] = np.mean(x_c, 0)  # mean of features of class c
            sigma[c, :] = np.std(x_c, 0)  # std of features of class c
        logging.info("Finish Fit")
        self.mu = mu  # C x D
        self.sigma = sigma  # C x D
        self.pi = (Nc + 1) / (
            N + C
        )  # Laplace smoothing (using alpha_c=1 for all c) you can derive using Dirichlet's distribution

        return self

    def logsumexp(self, Z):  # dimension C x N
        Zmax = np.max(Z, axis=0)[None, :]  # max over C
        log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
        return log_sum_exp

    def predict(self, xt):
        logging.info("Starting Predict")
        Nt, D = xt.shape
        # for numerical stability we work in the log domain
        # we add a dimension because this is added to the log-likelihood matrix
        # that assigns a likelihood for each class (C) to each test point, and so it is C x N
        log_prior = np.log(self.pi)[:, None]
        # logarithm of the likelihood term for Gaussian
        # the first two terms are the logarithm of the normalization term in the Gaussian and the final term is the exponent in the Gaussian.
        # Notice that we are adding dimensions (using None) to model parameters and data to make this evaluation.
        # The reason is that sigma and mu are C x D, while the data x is N x D. We operate on a C x N x D shape by increasing the number of dimensions when needed
        log_likelihood = (
            -0.5 * np.log(2 * np.pi)
            - np.log(self.sigma[:, None, :])
            - 0.5 * (((xt[None, :, :] - self.mu[:, None, :]) / self.sigma[:, None, :]) ** 2)
        )
        # now we sum over the feature dimension to get a C x N matrix (this has the log-likelihood for each class-test point combination)
        log_likelihood = np.sum(log_likelihood, axis=2)
        # posterior calculation
        log_posterior = log_prior + log_likelihood
        posterior = np.exp(log_posterior - self.logsumexp(log_posterior))
        return posterior.T  # dimension N x C