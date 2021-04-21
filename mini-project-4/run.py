#%%
import copy
import os
import time
import urllib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from helper import *

model = models.vgg19(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


# https://arxiv.org/pdf/1812.01187.pdf
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# train_dataset = datasets.ImageFolder(TRAIN_SET, preprocess)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = datasets.ImageFolder(VAL_SET, preprocess)
input_batch = torch.utils.data.DataLoader(val_dataset, shuffle=False)
print("data loaded")

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")


def validate(model, val_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    # lambd = 5e^-4
    print("starting validation")
    for i, data in enumerate(val_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        print(i, val_dataset.imgs[i])

        output = model(data)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        # plt.imshow(data.permute(1, 2, 0))
        print(data.shape)

        # loss = criterion(output, target) * lambd
        loss = criterion(output, target)
        print(loss)
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
        print("-------------------")

    val_loss = val_running_loss / len(val_dataloader.dataset)
    val_accuracy = 100.0 * val_running_correct / len(val_dataloader.dataset)
    return val_loss, val_accuracy


val_epoch_loss, val_epoch_accuracy = validate(model, input_batch)
print(val_epoch_loss, val_epoch_accuracy)
# %%
