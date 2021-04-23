import logging

import torch.nn as nn
import torchvision.models as models


def create_custom_model():
    model = models.vgg11(pretrained=True)

    model.features[1] = nn.LeakyReLU(inplace=True)
    model.features[4] = nn.LeakyReLU(inplace=True)
    model.features[7] = nn.LeakyReLU(inplace=True)
    model.features[9] = nn.LeakyReLU(inplace=True)
    model.features[14] = nn.LeakyReLU(inplace=True)
    model.features[17] = nn.LeakyReLU(inplace=True)
    model.features[19] = nn.LeakyReLU(inplace=True)

    model.classifier[1] = nn.LeakyReLU(inplace=True)
    model.classifier[4] = nn.LeakyReLU(inplace=True)
    # Reudcing the number of FC Layers
    # classifiers = [model.classifier[0], model.classifier[1], model.classifier[2], model.classifier[6]]
    # model.classifier = nn.Sequential(*classifiers)
    logging.info(
        f"""Model Config
    {model}
    """
    )
    return model
