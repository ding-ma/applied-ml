#%%
from PIL import Image

from helper import *

Image.open("/home/dataset/ILSVRC/Data/CLS-LOC/val/n04275548/ILSVRC2012_val_00043164.JPEG")


# %%

import torchvision.models as models
vgg = models.vgg16(pretrained=True)
print(vgg)