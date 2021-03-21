import joblib
from pathlib import Path
from model.AbstractMLP import AbstractMLP

pickle = "/home/ding/applied-ml/mini-project-3/pickles/2021-03-21_015655.pkl"
model:AbstractMLP = joblib.load(pickle)
