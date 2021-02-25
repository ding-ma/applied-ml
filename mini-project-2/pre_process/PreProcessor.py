from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer


dataset_path = Path("/home/dataset/project2")
dataset_path.mkdir(exist_ok=True, parents=True)

nltk.download("stopwords")
stemmer = SnowballStemmer("english")


bigram_series = pd.read_html(dataset_path.joinpath("bigrams.html"))[1]
bigram_series[0] = bigram_series[0].astype(str)
bigram_series[0] = bigram_series[0].apply(lambda x: x.replace(u"\xa0", "_"))

# make it a set because O(1) check vs O(n) check
bigram_series = set(bigram_series[0].to_list())


class PreProcessor:
    def __init__(
        self,
        df: pd.DataFrame,
        words_to_remove: list,
        f_name: str,
        process_column: str = "sentence",
        token: str = r"\w+",
    ):
        """df['sentence'] needs to be string """
        self.df = df
        self.stop_words = set(stopwords.words("english") + words_to_remove)
        self.f_name = f_name
        self.process_column = process_column
        self.tokenizer = RegexpTokenizer(token)
        self.__did_process = False
        self.exp_df = None

    def process(self) -> (pd.DataFrame, pd.DataFrame):
        """df['sentence'] is an array of tokenized words based on processing settings"""
        if self.__did_process:
            return self.df, self.exp_df

        ###################    cleaning    ###########################
        # add any other steps that you need. it will be applied to all dataset

        self.df[self.process_column] = self.df[self.process_column].astype(str)

        # remove \n, \r, \t into " "
        self.df[self.process_column] = self.df[self.process_column].apply(
            lambda x: x.replace("\n", " ").replace("\r", "").replace("\t", " ")
        )

        # tokenize the strings into array. makes them lowercase too
        self.df[self.process_column] = self.df[self.process_column].apply(lambda x: self.tokenizer.tokenize(x.lower()))

        # word stemming see: https://www.nltk.org/howto/stem.html
        self.df[self.process_column] = self.df[self.process_column].apply(lambda x: [stemmer.stem(i) for i in x])

        # compute 2 ngrams and check if they are in the list of ngrams
        self.df[self.process_column] = self.df[self.process_column].apply(
            lambda x: x + ["_".join(i) for i in list(ngrams(x, 2)) if "_".join(i) in bigram_series]
        )

        # remove common stopwords from ntlk library and also some domain specific words
        self.df[self.process_column] = self.df[self.process_column].apply(
            lambda x: [item for item in x if item not in self.stop_words]
        )

        #################################################################

        self.df.to_csv(
            dataset_path.joinpath("{}_row_array_stemmed.csv".format(self.f_name)),
            index=False,
        )

        # see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html
        self.exp_df = self.df.explode(self.process_column)
        self.exp_df.reset_index(inplace=True, drop=True)
        self.exp_df.to_csv(
            dataset_path.joinpath("{}_exploded_stemmed.csv".format(self.f_name)),
            index=False,
        )

        self.__did_process = True

        return self.df, self.exp_df
