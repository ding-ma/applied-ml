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
d = {}
files = Path('pickles').glob("*.pkl")
for file in files:
    model: AbstractMLP = joblib.load(file)
    d[file.stem + "_test_acc"] = model.test_acc_history
    d[file.stem + "_train_acc"] = model.train_acc_history
    d[file.stem + "_loss"] = model.loss_history

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
df.to_csv('all_runs.csv', index=False)

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

#%%
d = {}
files = Path('pickles').glob("*.pkl")
for file in files:
    model: AbstractMLP = joblib.load(file)
    d[file.stem + "_test_acc"] = model.test_acc_history
    d[file.stem + "_train_acc"] = model.train_acc_history
    d[file.stem + "_loss"] = model.loss_history

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
df.to_csv('all_runs.csv', index=False)

#%%
preprocess_param = {
    "threshold": False,
    "normalize": True,
    "augment_data": True,
}
train_array, y_train, test_array, y_test = aquire_data(**preprocess_param)

mlp: TwoLayer = joblib.load("/home/ding/applied-ml/mini-project-3/pickles/03-29_220042_two_layer_512_ReLU_256_ReLU_L2(True)_LR(0.002)_BS(50)_augmented_dataset(360k).pkl")
y_pred = mlp.predict(train_array)
mlp.file_name += "_train_plots"
mlp.plot(y_train, y_pred)


# %%
