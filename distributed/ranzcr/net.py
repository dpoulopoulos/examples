import torch.nn as nn
import torchvision.models as models

class ResNext(nn.Module):
    def __init__(self, pretrained=True, freeze=True, target_size: int = 11):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                     stride=(2, 2), padding=(3, 3), bias=False)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)

    def forward(self, x):
        return self.model(x)