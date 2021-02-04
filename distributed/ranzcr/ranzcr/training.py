import logging
import statistics

import torch
import torch.distributed as dist

from utils import get_score


logging.basicConfig(level=logging.DEBUG)


def train(args, model, device, train_loader, valid_loader, 
          optimizer, criterion, epoch, writer):
    scores = []
    
    logging.info(f"Started training with log interval: {args.log_interval}")

    model.train()
    for i, (data, target) in enumerate(train_loader):
        logging.info(f"Started training step {i}/{len(train_loader)}")
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss = {:.4f}'.format(
                  epoch, i, len(train_loader),
                  100. * i / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('loss', loss.item(), niter)

    logging.info(f"Starting evaluation")   
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(valid_loader):
            logging.info(f"Started evaluation step {i}/{len(valid_loader)}")
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            score = get_score(target.detach().cpu(), output.detach().cpu())
            scores.append(score)
            
        print('Train Epoch: {}\tScore = {:.4f}'.format(
              epoch, statistics.mean(scores)))

    # dist.barrier()
