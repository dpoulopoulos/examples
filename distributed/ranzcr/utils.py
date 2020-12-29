import os

from argparse import Namespace

import torch
import torch.nn as nn
import torch.distributed as dist

from sklearn.metrics import roc_auc_score


CHECKPOINT_PATH = "model.checkpoint"


def setup(rank: int, args: Namespace):
    """Initializes a distributed training process group.
    
    Args:
        rank: a unique identifier for the process
        args: user defined arguments
    """
    os.environ['MASTER_ADDR'] = args.host
    os.environ['MASTER_PORT'] = args.port

    # initialize the process group
    dist.init_process_group(backend=args.backend, 
                            init_method='env://', 
                            rank=rank, 
                            world_size=args.world_size)


def checkpoint(model: nn.Module, gpu: int, epoch: int):
    """Saves the model in master process and loads it everywhere else.
    
    Args:
        model: the model to save
        gpu: the device identifier
        epoch: the training epoch
    Returns:
        model: the loaded model
    """
    if gpu == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.module.state_dict(), CHECKPOINT_PATH + f"_{epoch}.pt")

    # use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    model.module.load_state_dict(
        torch.load(CHECKPOINT_PATH + f"_{epoch}.pt", map_location=map_location))

    return model


def get_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Calculates the AUC score.

    Args:
        labels: the ground truth
        predictions: the predictions
    Returns:
        score: the AUC score
    """
    score = roc_auc_score(labels.reshape(-1), predictions.reshape(-1))
    return score
