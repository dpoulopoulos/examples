import argparse

from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH     = Path("/opt/ranzcr/data/train")
TRAIN_DF_PATH = Path("/opt/ranzcr/data/sample.csv")

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
               'Swan Ganz Catheter Present']


class RanzcrDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 tfms: transforms.Compose = None):
        """Creates an RANZCR dataset.
    
        Args:
            df: the dataset as a pandas DataFrame
            dataset: train or test split
            tfms: a composition of image transforms
        """ 
        self.df = df
        self.file_names = self.df['StudyInstanceUID'].values
        self.labels = self.df[TARGET_COLS].values
        self.tfms = tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        file_path = f"{DATA_PATH}/{file_name}.jpg"
        image = Image.open(file_path)
        if self.tfms:
            augmented = self.tfms(image)
            image = augmented
        label = torch.tensor(self.labels[i]).float()
        return image, label


def load_data(frac: float = None) -> pd.DataFrame:
    """Loads the dataset in memory.
    
    Args:
        frac: if set returns a sample of the whole dataset
    Returns:
        data_df: the dataset as a pandas DataFrame
    """
    data_df = pd.read_csv(TRAIN_DF_PATH)

    if frac:
        return data_df.sample(frac=frac)

    return data_df


def split_data(df: pd.DataFrame, 
               train_size: float = .8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into train/valid.
    
    Args:
        df: the dataset as a pandas DataFrame
        train_size: the fraction of the train set size
    Returns:
        train_df: the train dataset as a pandas DataFrame
        valid_df: the valid dataset as a pandas DataFrame
    """
    groups = df['PatientID'].values
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=1)

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    for train_idx, test_idx in gss.split(df, groups=groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[test_idx].reset_index(drop=True)
        
    return train_df, valid_df


def get_transforms(image_size: int) -> transforms.Compose:
    """Creates a composition of image transformation functions.
    
    Args:
        image_size: the size of the image (hxw)
    Returns:
        tfms: a composition of image transforms
    """
    tfms = transforms.Compose([transforms.RandomResizedCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307), (0.3081))
                             ])
    return tfms


def create_loaders(train_df: pd.DataFrame, 
                   valid_df: pd.DataFrame,
                   args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Creates the train/valid dataloaders.
    
    Args:
        train_df: the training dataset as a pandas DataFrame
        valid_df: the validation dataset as a pandas DataFrame
        args: user defined arguments
    Returns:
        train_loader: the train dataloader
        valid_loader: the valid dataloader
    """
    _transforms = get_transforms(args.image_size)

    train_dataset = RanzcrDataset(train_df, tfms=_transforms)
    valid_dataset = RanzcrDataset(valid_df, tfms=_transforms)

    train_sampler = DistributedSampler(train_dataset, 
                                       num_replicas=args.world_size,
                                       rank=args.rank)
    valid_sampler = DistributedSampler(valid_dataset,
                                       num_replicas=args.world_size,
                                       rank=args.rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True, 
                              sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True, 
                              sampler=valid_sampler)
    
    return train_loader, valid_loader
