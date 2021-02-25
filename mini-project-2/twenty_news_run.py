import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from backports.zoneinfo import ZoneInfo
from model.CrossValidation import CrossVal
from model.Helpers import evaluate_acc, print_acc_err, DATASET_PATH, NAIVE_BAYES_REPEAT_DICT, LOGISITC_REPEAT_DICT
from  model.NaiveBayes import BernoulliBayes, MultiNomialBayes
import sys
from statistics import mean
import logging


MODEL = MultinomialNB

# only needed for kCV
VECTORIZER = TfidfVectorizer()

experiment_description = f"""
Testing with Stemmed words with bigrams, k=5 CV
MultiNomialBayes(), TfidfVectorizer()
"""

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename="logs/News-{}.log".format(datetime.now().strftime("%Y-%m-%d_%H%M%S"))),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info(experiment_description)

twenty_news_df = pd.read_csv(DATASET_PATH.joinpath("twenty_news_row_array_bigram_stemmed.csv"))
twenty_news_df = shuffle(twenty_news_df, random_state=1)
twenty_news_df["sentence"] = twenty_news_df["sentence"].apply(lambda x: " ".join(eval(x)))

twenty_news_df_X = twenty_news_df["sentence"]
twenty_news_df_y = twenty_news_df["target"]

twenty_CV = CrossVal(twenty_news_df_X, twenty_news_df_y)
res = twenty_CV.kfoldCV(MultiNomialBayes(), TfidfVectorizer())
print_acc_err(res)