import statistics
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from net import create_model
from data import split_data, create_loaders, get_test_data
from utils import setup, cleanup, checkpoint, get_score, load_data


def train(gpu: int,args: Namespace):
    """Implements the training loop for PyTorch a model.

    Args:
        gpu: the GPU device
        args: user defined arguments
    """
    
    # global variables for evaluation
    losses = []
    scores = []

    # setup process groups
    rank = args.nr * args.gpus + gpu
    setup(rank, args)
    
    # define the model
    model = create_model()
    model.cuda(gpu)
    # Wrap the model
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), args.lr)

    # split data
    data_df = load_data(frac=.1)
    train_df, valid_df = split_data(data_df, args.train_size)
    train_loader, valid_loader = create_loaders(train_df, valid_df, rank, args) 

    if gpu == 0:
        print(f"Training started") 

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
        model = checkpoint(model, gpu, epoch)
            
    if args.save_model and gpu == 0:
        torch.save(model.module.state_dict(), "model.pt")
        
    cleanup()

    
def test(model_path):
    probs = []
    
    test_loader = get_test_data()
    
    model = create_model()
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
