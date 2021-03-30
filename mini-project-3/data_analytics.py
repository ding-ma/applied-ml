#%%
from model import *
from utils.activation import *

import sys
from pathlib import Path
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import logging
import sys
from datetime import datetime
import inspect

from preprocess.get_data import aquire_data
from itertools import islice
import numpy as np
import joblib


preprocess_param = {
    "threshold": False,
    "normalize": True,
    "augment_data": False,
}
_, y_train, _, y_test = aquire_data(**preprocess_param)

train = pd.DataFrame({
    "label":y_train
}).value_counts().to_frame("counts")
train['train_or_test'] = "train"

test = pd.DataFrame({
    "label":y_test
}).value_counts().to_frame("counts")
test['train_or_test'] = "test"

combined = pd.concat([train,test])
combined['labels'] = combined.index
combined['labels'] = combined['labels'].apply(lambda x: x[0])

sns.barplot(x="labels", y="counts", hue="train_or_test", data=combined)
plt.legend(loc=(1.04,0), title="Train/Test Counts")
plt.title("MNIST Data Distribution")
# %%
