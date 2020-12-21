from argparse import Namespace

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from .net import ResNext
from .data import get_data


def train(gpu: int, writer: SummaryWriter, args: Namespace):
    """Implements the training loop for PyTorch a model.

    Args:
        gpu: the GPU device
        writer: TensorBoard logging
        args: user defined arguments
    """

    # define the hyperameters
    BATCH_SIZE = args.batch
    LR = args.lr
    EPOCHS = args.epochs

    # setup process groups
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend=args.backend, init_method='env://',
                            world_size=args.world_size, rank=rank)
    
    # define the model
    torch.cuda.set_device(gpu)

    model = ResNext()
    model.cuda(gpu)
    # Wrap the model
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = Adam(model.parameters(), LR)

    # data loading
    train_loader = get_data(args, rank=rank)

    model.train()
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch, i * len(images), len(train_loader.dataset),
                    100. * i / len(train_loader), loss.item()))
                n_iter = epoch * len(train_loader) + i
                writer.add_scalar('loss', loss.item(), n_iter)


# def test(args: Namespace, model: nn.Module,
#          device: str, test_loader: DataLoader, 
#          loss_fn: nn.Module, epoch: int,
#          writer: SummaryWriter):
#     """Implements the evaluation loop for a PyTorch model.

#     Args:
#         gpu: the GPU device
#         args: user defined arguments
#     """
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
#     writer.add_scalar('accuracy', float(correct) / len(test_loader.dataset), epoch)
