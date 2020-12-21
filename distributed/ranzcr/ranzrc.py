import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from .utils import train


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default False)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model (default false)')
    parser.add_argument('--dir', default='logs',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--nodes', type=int, default=1,
                        help='number of nodes (default: 1)')
        parser.add_argument('--nr', type=int, default=0,
                        help='ranking within the nodes')
        parser.add_argument('--gpus', type=int, default=1,
                        help='number of gpus per node (default: 1)')
        parser.add_argument('--backend', type=str, help='distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL],
                            default=dist.Backend.GLOO)
        parser.add_argument('--host', type=str, default="localhost",
                        help='master address')
        parser.add_argument('--port', type=str, default="5000",
                            help='master port')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
    
    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    if dist.is_available():
        os.environ['MASTER_ADDR'] = args.host
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus * args.nodes
        mp.spawn(train, nprocs=args.gpus, args=(writer, args,))
