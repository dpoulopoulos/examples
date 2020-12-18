import os
import logging
import argparse

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from net import Net
from utils import get_data


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='G',
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='E',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='the learning rate')
    parser.add_argument('--batch', default=32, type=int, metavar='B',
                        help='the batch size')
    parser.add_argument('--conf', default="single", type=str, metavar='C',
                        help='distributed training configuration')
    parser.add_argument('--backend', default="gloo", type=str, metavar='D',
                        help='distributed training backend')
    parser.add_argument('--host', default="localhost", type=str, metavar='H',
                        help='master address')
    parser.add_argument('--port', default="5000", type=str, metavar='P',
                        help='master port')
    args = parser.parse_args()

    conf = args.conf
    if conf not in ["single", "dp", "ddp"]:
        raise ValueError(f"Configuration {conf} is not supported."
                         f" Please choose one of 'single', 'dp', or 'ddp'.")

    backend = args.backend
    if backend not in ["gloo", "nccl"]:
        raise ValueError(f"Backend {backend} is not supported."
                         f" Please choose one of 'gloo' or 'nccl'.")

    if args.gpus > 1:
        if conf == "ddp":
            os.environ['MASTER_ADDR'] = args.host
            os.environ['MASTER_PORT'] = args.port
            args.world_size = args.gpus * args.nodes
            mp.spawn(ddp_train, nprocs=args.gpus, args=(args,))
        else:
            train(args, distribute=True)
    else:
        train(args)


def ddp_train(gpu, args):
    logging.info(f"Training Process started on GPU: {gpu}")

    # define the hyperameters
    batch_size = args.batch
    lr = args.lr
    epochs = args.epochs

    # setup process groups
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend=args.backend, init_method='env://',
                            world_size=args.world_size, rank=rank)

    # define the model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    model = Net()
    model.cuda(gpu)
    # Wrap the model
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    # data loading
    train_loader, test_loader = get_data(args, distribute=True, rank=rank)
    total_step = len(train_loader)

    # training loop
    start = datetime.now()
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
        print("Training completed in: " + str(datetime.now() - start))
        
    # evaluation loop
    start = datetime.now()
        
    total = 0
    test_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item() 
            pred = outputs.argmax(dim=1, keepdim=True)
            total += labels.size(0)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    test_loss /= total

    if gpu == 0:
        print(f"Test set: Average loss: {test_loss:.4f},"
              f" Accuracy: {correct}/{total}"
              f" ({100. * correct / total:.0f}%)")
        
        logging.info("Evaluation completed in: " + str(datetime.now() - start))

    dist.destroy_process_group()


def train(args, distribute=False):
    if distribute:
        logging.info(f"Training Process started on {args.gpus} GPU(s)")
    else:
        logging.info("Training process started on 1 GPU")

    # define hyperparameters
    batch_size = args.batch
    lr = args.lr
    epochs = args.epochs

    # define the model
    model = Net().cuda()
    if distribute:
        model = nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    # data loading
    train_loader, test_loader = get_data(args)
    total_step = len(train_loader)

    # training loop
    start = datetime.now()
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

    logging.info("Training completed in: " + str(datetime.now() - start))
    
    # evaluation loop
    start = datetime.now()
        
    total = 0
    test_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item() 
            pred = outputs.argmax(dim=1, keepdim=True)
            total += labels.size(0)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    test_loss /= total

    print(f"Test set: Average loss: {test_loss:.4f},"
          f" Accuracy: {correct}/{total}"
          f" ({100. * correct / total:.0f}%)")
    
    logging.info("Evaluation completed in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()
