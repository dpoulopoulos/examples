from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import GroupKFold


TRAIN_PATH  = Path("data/train")
TEST_PATH   = Path("data/test")
DATA_PATH   = Path("data/train.csv")
TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
               'Swan Ganz Catheter Present']


class RanzcrDataset(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.file_names = self.df['StudyInstanceUID'].values
        self.labels = self.df[TARGET_COLS].values
        self.tfms = tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        file_path = f"{TRAIN_PATH}/{file_name}.jpg"
        image = Image.open(file_path)
        if self.tfms:
            augmented = self.tfms(image)
            image = augmented
        label = torch.tensor(self.labels[i]).float()
        return image, label
    

def get_data(args, df, fold, rank):
    trn_idx = df[df['Fold'] != fold].index
    val_idx = df[df['Fold'] == fold].index
    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)

    _transforms = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train_dataset = RanzcrDataset(train_folds, _transforms)
    valid_dataset = RanzcrDataset(valid_folds, _transforms)

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


def split_data(folds):
    folds_df = pd.read_csv(DATA_PATH)
    groups = folds_df['PatientID'].values

    for n, (train_index, val_index) in enumerate(GroupKFold(folds).split(folds_df, folds_df[TARGET_COLS], groups)):
        folds_df.loc[val_index, 'Fold'] = int(n)

    folds_df['Fold'] = folds_df['Fold'].astype("int")

    return folds_df
    