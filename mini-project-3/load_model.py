#%%
import joblib
from pathlib import Path
from model.AbstractMLP import AbstractMLP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess.get_data import aquire_data
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

pickle = "/home/ding/applied-ml/mini-project-3/pickles/03-21_204649_two_layer_512_ReLU_256_ReLU_L2(True)_LR(0.0002)_BS(10).pkl"
model: AbstractMLP = joblib.load(pickle)
tmp = pd.DataFrame({
    "loss": model.loss_history
})
tmp['avg'] = tmp.rolling(window=4).mean()
tmp['diff'] = (tmp['loss']-tmp['avg']).abs()