#%%
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class DataWrapper(Dataset):
    def __init__(self, img_path, mapper_path, transform=None):
        self.imgs = list(Path(img_path).rglob("*.JPEG"))
        self.mapper = json.load(open(mapper_path))
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.imgs[idx]
        label, _ = self.mapper[img_path.parents[0].name]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return (image, torch.tensor([int(label)]))

    def __len__(self):
        return len(self.imgs)


# %%
