import statistics
from argparse import Namespace

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from net import ResNext
from data import split_data, get_data
from utils import setup, cleanup, checkpoint, get_score


def train(gpu: int, args: Namespace):
    """Implements the training loop for PyTorch a model.

    Args:
        gpu: the GPU device
        args: user defined arguments
    """

    # setup process groups
    rank = args.nr * args.gpus + gpu
    setup(rank, args)
    
    # define the model
    model = ResNext()
    model.cuda(gpu)
    # Wrap the model
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), args.lr)

    # split data
    train_df = split_data(args.folds)

    for fold in range(args.folds):
        losses = []
        scores = []
        train_loader, valid_loader = get_data(args, train_df, fold, rank)
        
        # checkpoint model
        model = checkpoint(model, gpu)
        
        if gpu == 0:
            print(f"Training started using fold {fold} for validation") 
        
        # train
        model.train()
        for epoch in range(args.epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.cuda(gpu)
                labels = labels.cuda(gpu)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % args.log_interval == 0 and gpu ==0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                          epoch+1, i, len(train_loader),
                          100. * i / len(train_loader), loss.item()))
        
        # evaluate
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.cuda(gpu)
                labels = labels.cuda(gpu)
                output = model(images)
                loss = criterion(output, labels).item()
                score = get_score(labels.detach().cpu(), output.detach().cpu())
                losses.append(loss)
                scores.append(score)

            if gpu == 0:
                print("Validation loss={:.4f}\tAUC score={:.4f}".format(
                      statistics.mean(losses), statistics.mean(scores)))
            
    if args.save_model and gpu == 0:
        torch.save(model.state_dict(), "model.pt")
        
    cleanup()

    
# def test(args: Namespace, model: nn.Module):
#     """Implements the evaluation loop for a PyTorch model.

#     Args:
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
