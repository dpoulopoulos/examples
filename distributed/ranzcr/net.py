import torch.nn as nn
import torchvision.models as models


def create_model(pretrained=True, freeze=True, target_size: int = 11):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, target_size)
    
    return model
