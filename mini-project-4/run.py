#%%
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from helper import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# https://arxiv.org/pdf/1812.01187.pdf
normalize = transforms.Normalize(mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375])

train_dataset = datasets.ImageFolder(
    TRAIN_SET,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

val_dataset = datasets.ImageFolder(
    VAL_SET,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

vgg16 = models.vgg16(pretrained=True)
vgg16.to(device)

print(vgg16)

criterion = nn.MSELoss()


def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100.0 * train_running_correct / len(train_dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")


def validate(model, val_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    # lambd = 5e^-4

    for int, data in enumerate(val_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        # loss = criterion(output, target) * lambd
        loss = criterion(output, target)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / len(val_dataloader.dataset)
    val_accuracy = 100.0 * val_running_correct / len(val_dataloader.dataset)

    return val_loss, val_accuracy


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit(vgg16, train_dataloader)
    val_epoch_loss, val_epoch_accuracy = validate(vgg16, val_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
print((end - start) / 60, "minutes")


#%%
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color="green", label="train accuracy")
plt.plot(val_accuracy, color="blue", label="validataion accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

#%%
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color="orange", label="train loss")
plt.plot(val_loss, color="red", label="validataion loss")
plt.legend()
plt.savefig("loss.png")
plt.show()
