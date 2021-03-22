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


d = {}
files = Path('pickles').glob("*.pkl")
for file in files:
    model: AbstractMLP = joblib.load(file)
    d[file.stem + "_test_acc"] = model.test_acc_history
    d[file.stem + "_train_acc"] = model.train_acc_history
    d[file.stem + "_loss"] = model.loss_history

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
df.to_csv('all_runs.csv', index=False)