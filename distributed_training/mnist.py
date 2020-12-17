import os
import logging
import argparse

from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--mode', default="single", type=str, metavar='D',
                        help='enable distributed training')
    args = parser.parse_args()
    
    if args.mode == "ddp" and args.gpus > 1:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5555'
        mp.spawn(ddp_train, nprocs=args.gpus, args=(args,))
    elif args.mode == "dp" and args.gpus > 1:
        dp_train(args)
    else:
        train(args)


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
        
        return F.log_softmax(x, dim=1)


def ddp_train(gpu, args): 
    logging.info(f"Process started on GPU: {gpu}")
    
    batch_size = 32
    learning_rate = 1e-3
    epochs = args.epochs
    
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    model = Net()
    model.cuda(gpu)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
   
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root="data",
                                               train=True,
                                               transform=transform,
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    total_step = len(train_loader)
    
    start = datetime.now()
    logging.info(f"Training started on GPU: {gpu}")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        
    dist.destroy_process_group()
    
    
def dp_train(args):
    logging.info(f"Process started on {torch.cuda.device_count()} GPU(s)")
    
    batch_size = 128
    learning_rate = 1e-3
    epochs = args.epochs
    
    model = Net().cuda()
    model = nn.DataParallel(model)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root="data",
                                               train=True,
                                               transform=transform,
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    total_step = len(train_loader)
    
    start = datetime.now()
    logging.info(f"Training started on {torch.cuda.device_count()} GPU(s)")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    
    print("Training complete in: " + str(datetime.now() - start))
    
def train(args):
    logging.info(f"Process started on 1 GPU")
    
    batch_size = 32
    learning_rate = 1e-3
    epochs = args.epochs
    
    model = Net().cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root="data",
                                               train=True,
                                               transform=transform,
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    total_step = len(train_loader)
    
    start = datetime.now()
    logging.info(f"Training started on 1 GPU")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    
    print("Training complete in: " + str(datetime.now() - start))

    
if __name__ == '__main__':
    main()