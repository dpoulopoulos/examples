import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


def get_transforms() -> transforms:
    _transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return _transforms


def create_loader(dataset: Dataset, batch_size:int = 32,
                  sampler: Sampler = None) -> DataLoader:
    if sampler:
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True,
                          sampler=sampler)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=0,
                      pin_memory=True)


def get_data(args, distribute: str = False, rank: int = 0):
    _transforms = get_transforms()
    
    train_dataset = MNIST(root="data",
                          train=True,
                          transform=_transforms,
                          download=True)

    test_dataset = MNIST('data', train=False, transform=_transforms)

    if distribute:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=args.world_size,
                                           rank=rank)
        test_sampler = DistributedSampler(test_dataset,
                                           num_replicas=args.world_size,
                                           rank=rank)

        return create_loader(train_dataset, args.batch, train_sampler), \
               create_loader(test_dataset, args.batch, test_sampler)

    return create_loader(train_dataset, args.batch), create_loader(test_dataset, args.batch)
