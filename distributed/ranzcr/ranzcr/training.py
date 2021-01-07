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


def train(args, model, device, train_loader, valid_loader, optimizer, epoch, writer):
    scores = []
    
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss = {:.4f}'.format(
                  epoch, i * len(data), len(train_loader),
                  100. * i / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('loss', loss.item(), niter)
            
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            score = get_score(labels.detach().cpu(), output.detach().cpu())
            scores.append(score)
            
        print('Train Epoch: {}\tScore = {:.4f}'.format(
              epoch, statistics.mean(scores)))