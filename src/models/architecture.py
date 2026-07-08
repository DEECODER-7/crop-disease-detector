"""
Model architecture: ResNet18 pretrained on ImageNet, with the final
classification layer replaced to output our number of crop-disease classes.

Why ResNet18 specifically: it's small enough to train reasonably on CPU/laptop
GPUs, well-understood/widely used (so it's a safe, defensible interview answer
rather than an exotic choice you can't explain), and strong as a transfer-learning
base for image classification tasks like this one.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    freeze_backbone=True: only train the new final layer, keep all the
    pretrained convolutional weights frozen. This is the standard first move
    for transfer learning — the pretrained backbone already knows how to
    detect edges/textures/shapes from ImageNet, we just need to teach it to
    map those features to OUR classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer. This new layer is created with
    # requires_grad=True by default, so even with the backbone frozen, this
    # layer WILL be trained — that's the whole point.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
