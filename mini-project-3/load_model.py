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

pickle = "/home/ding/applied-ml/mini-project-3/pickles/2021-03-21_162719_two_layer_128_ReLU_128_ReLU_L2(False)_LR(0.0005)_BT(5).pkl"
model:AbstractMLP = joblib.load(pickle)

epochs = [x[0] for x in model.loss_history]
loss = [x[1] for x in model.loss_history]

train = [x[1] for x in model.train_acc_history]

test = [x[1] for x in model.test_acc_history]

df = pd.DataFrame({
    "loss": loss,
    # "train": train,
    # "test": test
})

sns.lineplot(data=df)
plt.xlabel("epochs")
plt.ylabel("loss")

# %%
df = pd.DataFrame({
    "train": train,
    "test": test
})

sns.lineplot(data=df)
plt.xlabel("epochs")
plt.ylabel("Acc")
plt.title("Train and Test Accuracy over Epochs")