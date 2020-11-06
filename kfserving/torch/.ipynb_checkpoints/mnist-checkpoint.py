#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms


# In[2]:


LR = 1e-3
BATCH_SIZE = 500
EPOCHS = 4


# In[3]:


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# In[4]:


x_trn = datasets.MNIST('data', train=True, transform=transform, download=True)
x_tst = datasets.MNIST('data', train=False, transform=transform)


# In[5]:


trn_loader = torch.utils.data.DataLoader(x_trn, batch_size=BATCH_SIZE)
tst_loader = torch.utils.data.DataLoader(x_tst, batch_size=BATCH_SIZE)


# In[6]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# In[7]:


model = Net()
optimizer = optim.Adam(model.parameters(), lr=LR)


# In[8]:


log_interval = 40
model.train()

for epoch in range(1, EPOCHS + 1):
    for idx, (data, target) in enumerate(trn_loader):
        data, target = data, target
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{idx * len(data)}/{len(trn_loader.dataset)}"
                  f" ({100. * idx / len(trn_loader):.0f}%)]\tLoss: {loss.item():.6f}")


# In[9]:


tst_loss = 0
correct = 0
model.eval()

with torch.no_grad():
    for data, target in tst_loader:
        data, target = data, target
        output = model(data)
        tst_loss += F.nll_loss(output, target, reduction='sum').item() 
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
tst_loss /= len(tst_loader.dataset)

print(f"\nTest set: Average loss: {tst_loss:.4f},"
      f" Accuracy: {correct}/{len(tst_loader.dataset)}"
      f" ({100. * correct / len(tst_loader.dataset):.0f}%)\n")


# In[12]:


torch.save(model.state_dict(), "model.pt")


# In[ ]:




