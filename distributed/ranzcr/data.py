import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class RanzcrDataset(Dataset):
    def __init__(self, df, target_cols, path, tfms=None):
        self.path = path
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[target_cols].values
        self.tfms = tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.path}/{file_name}.jpg'
        image = Image.open(file_path)
        if self.tfms:
            augmented = self.tfms(image)
            image = augmented
        label = torch.tensor(self.labels[idx]).float()
        return image, label
    

def get_data(folds, target_cols, path, batch_size: int = 128, 
             shuffle: bool = True, drop_last: bool = True):
    _transforms = transforms.Compose([transforms.RandomResizedCrop(512),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                     ])
    dataset = RanzcrDataset(folds, target_cols, path, _transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=0,
                            pin_memory=True, drop_last=drop_last)
    return dataloader
    
    