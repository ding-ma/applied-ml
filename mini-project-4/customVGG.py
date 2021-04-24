import logging

import torch.nn as nn
import torchvision.models as models


# python main.py /home/dataset/ILSVRC/Data/CLS-LOC --arch custom --keep-logs --batch-size 175
def create_custom_model():
    model = models.vgg11(pretrained=False)

    # remove the last maxpool and conv layers
    model.features = nn.Sequential(*[model.features[i] for i in range(16)])

    logging.info(
        f"""Model Config
    {model}
    """
    )
    return model
