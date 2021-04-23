import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets


class JitterDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, jitter):
        self.dataset = dataset
        self.jitter = jitter
        self.max = max(jitter)
        for jit in jitter:
            if jit % 2 != 0:
                raise ValueError("Jitter values must be multiple of two")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i = self.jitter[idx % len(self.jitter)]

        transform = transforms.Compose(
            [
                transforms.Resize(i),
                transforms.CenterCrop(i),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        data, target = self.dataset[idx]
        data = transform(data)
        resize = (self.max - i) // 2
        padded_img = F.pad(input=data, pad=(resize, resize, resize, resize), mode="constant", value=0)
        return padded_img, target
