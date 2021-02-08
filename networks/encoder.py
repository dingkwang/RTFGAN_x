
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        resnet = models.resnet34(pretrained=True)                                                             
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.layer = nn.Sequential(nn.Linear(resnet.fc.in_features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.layer(self.resnet(x))
