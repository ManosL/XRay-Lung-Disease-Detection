from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models

class ResNetType(Enum):
    RESNET18  = 0
    RESNET34  = 1
    RESNET50  = 2
    RESNET101 = 3
    RESNET152 = 4



class ResNetXRayModel(nn.Module):
    # Remember to convert the image to RGB form, because ResNet is not
    # trained on greyscale
    def __init__(self, resnet_type, pretrained=True, layers_to_freeze=[]):
        super().__init__()
        assert(isinstance(resnet_type, ResNetType))

        # Extracting the appropriate resnet model and its corresponding
        # number of features
        resnet_model = None
        features_num = None

        if resnet_type == ResNetType.RESNET18:
            resnet_model = models.resnet18(pretrained=pretrained)
            features_num = 512
        elif resnet_type == ResNetType.RESNET34:
            resnet_model = models.resnet34(pretrained=pretrained)
            features_num = 512
        elif resnet_type == ResNetType.RESNET50:
            resnet_model = models.resnet50(pretrained=pretrained)
            features_num = 2048
        elif resnet_type == ResNetType.RESNET101:
            resnet_model = models.resnet101(pretrained=pretrained)
            features_num = 2048
        elif resnet_type == ResNetType.RESNET152:
            resnet_model = models.resnet152(pretrained=pretrained)
            features_num = 2048

        # Freeze some layers(we do not freeze the conv1 layer because
        # from it we will probably get our initial features)
        resnet_layers = [resnet_model.layer1, resnet_model.layer2,
                        resnet_model.layer3, resnet_model.layer4]
        
        for layer_index in layers_to_freeze:
            resnet_layer = resnet_layers[layer_index]

            # Freeze the layer
            for param in resnet_layer.parameters():
                param.requires_grad = False

        # Extract all the layers except the FC one
        self.resnet_feature_extractor = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))

        # Initialize our classification layers
        self.clf = None

        if features_num == 2048:
            self.clf = torch.nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(features_num, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 4),
                            nn.Softmax(dim=1)          
                        )
        elif features_num == 512:
            self.clf = torch.nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(features_num, 4),
                            nn.Softmax(dim=1)          
                        )


        assert(self.clf != None)

        return
    
    def forward(self, x):
        features = self.resnet_feature_extractor(x)
        pred     = self.clf(features)

        return pred