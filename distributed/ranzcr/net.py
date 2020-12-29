import torch.nn as nn
import torchvision.models as models


def create_model(pretrained: bool = True, 
                 freeze: bool = True,
                 channels_in: int = 1,
                 output_size: int = 11) -> nn.Module:
    """Creates and initializes a Neural Network.
    
    Args:
        pretrained: whether to use transfer learning
        freeze: freeze the pretrained layers
        channels_in: the number of image channels 
        output_size: number of nodes in the output layer
    """
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(channels_in, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    features_in = model.fc.in_features
    model.fc = nn.Linear(features_in, output_size)
    
    return model
