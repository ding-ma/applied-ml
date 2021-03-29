#%%
import joblib
from pathlib import Path
from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess.get_data import aquire_data
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

#%%
ten:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-21_221051_two_layer_256_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(10)_n_train(10000).pkl")
twenty:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-21_221051_two_layer_256_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(10)_n_train(20000).pkl")
thirty:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-21_221051_two_layer_256_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(10)_n_train(30000).pkl")
fourty:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-21_221051_two_layer_256_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(10)_n_train(40000).pkl")

df = pd.DataFrame({
    "10k": ten.test_acc_history,
    "20k": twenty.test_acc_history,
    "30k": thirty.test_acc_history,
    "40k": fourty.test_acc_history
})

sns.lineplot(data=df)
plt.xlabel("Epochs")
plt.ylabel("Test accuracy")
plt.title("Test accuracy with different training size")
plt.legend(title="Size of training set")
plt.xticks(range(0,19, 3))

#%%

big: TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-21_204649_two_layer_512_ReLU_256_ReLU_L2(True)_LR(0.0002)_BS(10).pkl")
smaller:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-22_014532_two_layer_256_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(10).pkl")
same:TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-22_020846_two_layer_128_ReLU_128_ReLU_L2(True)_LR(0.0002)_BS(50).pkl")
d = {
    "512->256": big.test_acc_history,
    "256->128": smaller.test_acc_history,
    "128->128": same.test_acc_history
}

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

sns.lineplot(data=df[:10])
# %%
df = pd.read_csv("/home/ding/applied-ml/mini-project-3/all_runs.csv")
sns.lineplot(data=df)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
