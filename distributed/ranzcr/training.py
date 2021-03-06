import statistics
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from net import ResNext
from data import split_data, get_data, get_test_data
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
    model = ResNext().architecture
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

                if i % args.log_interval == 0 and gpu == 0:
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
                
        # checkpoint model
        model = checkpoint(model, gpu, fold)
            
    if args.save_model and gpu == 0:
        torch.save(model.module.state_dict(), "model.pt")
        
    cleanup()

    
def test(model_path):
    probs = []
    
    test_loader = get_test_data()
    
    model = ResNext().architecture
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            avg_preds = []
            images = images.cuda()
            preds = model(images)
            avg_preds.append(preds.sigmoid().detach().to('cpu').numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probs.append(avg_preds)
        probs = np.concatenate(probs)
    return probs