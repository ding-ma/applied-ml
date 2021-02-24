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
from model.CrossValidation import CrossVal
from model.Helpers import evaluate_acc, print_acc_err, dataset_path
import sys
from statistics import mean
import logging


experiment_description = """
in this we ran.....

"""


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename="logs/twenty_news-{}.log".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))),
        logging.StreamHandler(sys.stdout),
    ],
)


logging.info(experiment_description)

twenty_news_df = pd.read_csv("dataset/twenty_news_row_array_bigram.csv")
twenty_news_df = shuffle(twenty_news_df, random_state=1)
twenty_news_df["sentence"] = twenty_news_df["sentence"].apply(lambda x: " ".join(eval(x)))

twenty_news_df_X = twenty_news_df["sentence"]
twenty_news_df_y = twenty_news_df["target"]

twenty_CV = CrossVal(twenty_news_df_X, twenty_news_df_y)
res = twenty_CV.kfoldCV(MultinomialNB(), CountVectorizer())
print_acc_err(res)