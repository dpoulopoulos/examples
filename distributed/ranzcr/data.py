from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


TRAIN_PATH  = Path("data/train")
DATA_PATH   = Path("data/train.csv")
TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
               'Swan Ganz Catheter Present']


class RanzcrDataset(Dataset):
    def __init__(self, tfms=None):
        self.df = pd.read_csv(DATA_PATH)
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
    

def get_data(args, rank):
    _transforms = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                     ])
    dataset = RanzcrDataset(_transforms)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size,
                                 rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True, sampler=sampler)
    return dataloader
    