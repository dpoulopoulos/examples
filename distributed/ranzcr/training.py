import time
import statistics

from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from net import create_model
from data import load_data, split_data, create_loaders, create_test_loaders
from utils import setup, cleanup, checkpoint, get_score


def train(gpu: int, args: Namespace):
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
    if args.fine_tune:
        if gpu == 0:
            print("Loading an existing model")
        model = create_model(freeze=False)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load("model.pt", map_location=map_location))
    else:
        model = create_model()
    model.cuda(gpu)
    # Wrap the model
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # split data
    if args.sample:
        data_df = load_data(frac=args.sample)
    else:
        data_df = load_data()
    train_df, valid_df = split_data(data_df, args.train_size)
    train_loader, valid_loader = create_loaders(train_df, valid_df, rank, args) 

    if gpu == 0:
        print("Training started")
        print(f"Training set: {len(train_df)}\tValidation set: {len(valid_df)}")

    # train
    model.train()
    for epoch in range(args.epochs):
        LOSS = 0.
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(gpu)
            labels = labels.cuda(gpu)
            output = model(images)
            loss = criterion(output, labels)
            LOSS += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % args.log_interval == 0 and gpu == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                      epoch+1, i, len(train_loader),
                      100. * i / len(train_loader), loss.item()))
        
        if gpu == 0:
            print(f"Average loss for epoch {epoch+1}: {LOSS / len(train_loader)}")
            
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
        checkpoint(model, gpu, epoch)
            
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
