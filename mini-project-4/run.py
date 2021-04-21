import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from helper import *


# https://arxiv.org/pdf/1812.01187.pdf
normalize = transforms.Normalize(mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375])

test_dataset = datasets.ImageFolder(
    TEST_SET,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
