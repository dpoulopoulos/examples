from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import GroupShuffleSplit


TRAIN_PATH    = Path("data/train")
TEST_PATH     = Path("data/test")
TRAIN_DF_PATH = Path("data/train.csv")
TEST_DF_PATH  = Path("data/sample_submission.csv")

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
               'Swan Ganz Catheter Present']


class RanzcrDataset(Dataset):
    def __init__(self, df, dataset="train", tfms=None):
        self.df = df
        self.dataset = dataset
        self.file_names = self.df['StudyInstanceUID'].values
        self.labels = self.df[TARGET_COLS].values
        self.tfms = tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        if self.dataset == "train":
            file_path = f"{TRAIN_PATH}/{file_name}.jpg"
        else:
            file_path = f"{TEST_PATH}/{file_name}.jpg"
        image = Image.open(file_path)
        if self.tfms:
            augmented = self.tfms(image)
            image = augmented
        label = torch.tensor(self.labels[i]).float()
        return image, label


def get_data(args, train_df, valid_df, rank):
    _transforms = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train_dataset = RanzcrDataset(train_df, tfms=_transforms)
    valid_dataset = RanzcrDataset(valid_df, tfms=_transforms)

    train_sampler = DistributedSampler(train_dataset, 
                                       num_replicas=args.world_size,
                                       rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, 
                                       num_replicas=args.world_size,
                                       rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  pin_memory=False, sampler=train_sampler)

    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  pin_memory=False, sampler=valid_sampler)
    
    return train_dataloader, valid_dataloader


def get_test_data():
    test_df = pd.read_csv(TEST_DF_PATH)
    
    _transforms = transforms.Compose([transforms.RandomResizedCrop(512),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    test_dataset = RanzcrDataset(test_df, "test", tfms=_transforms)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    return test_loader


def split_data(train_size=.8):
    train_df = pd.read_csv(TRAIN_DF_PATH)
    groups = train_df['PatientID'].values
    
    gss = GroupShuffleSplit(n_splits=2, train_size=train_size, random_state=1)

    for train_idx, test_idx in gss.split(train_df, groups=groups):
        train = train_df.loc[train_idx].reset_index(drop=True)
        valid = train_df.loc[test_idx].reset_index(drop=True)

    return train, valid
    