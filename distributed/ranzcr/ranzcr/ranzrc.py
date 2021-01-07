import os
import logging
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from training import train
from net import create_model
from utils import should_distribute
from data import load_data, split_data, create_loaders


logging.basicConfig(level=logging.INFO)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32,
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='l2 regularization (default: 0.1)')
    parser.add_argument('--train-size', type=int, default=.8,
                        help='training set split (default: 0.8)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='the size of the image (default: 224)')
    parser.add_argument('--sample', type=float, default=None,
                        help='load a dataset sample (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.NCCL)
        
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logging.info('Using CUDA')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    if should_distribute():
        logging.info('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    
    df = load_data(args.sample)
    train_df, valid_df = split_data(df, args.train_size)
    train_loader, valid_loader = create_loaders(train_df, valid_df, args)

    model = create_model()
    model = model.to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                           weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)

    if (args.save_model):
        torch.save(model.state_dict(), "ranzcr_model.pt")
        
if __name__ == '__main__':
    main()