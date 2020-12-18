import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.max_pool(x)
        x = self.dropout1(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return self.log_softmax(x)