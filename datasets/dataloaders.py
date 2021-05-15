##################################################
# Imports
##################################################

from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


class TransformDataset(Dataset):
    def __init__(self, ds, transform=None, target_transform=None):
        super().__init__()
        self.ds = ds
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds.__getitem__(idx)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def get_transforms(args):
    if args.backbone in ['vit', 'mlp_mixer']:
        transform = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'train_aug': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'validation': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }
    else:
        transform = {
            'train': transforms.ToTensor(),
            'train_aug': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'validation': transforms.ToTensor(),
            'test': transforms.ToTensor(),
        }
    return transform

def get_datasets(args):

    # Transforms
    transform = get_transforms(args)

    # Dataset
    ds_args = {
        'root': args.data_base_path,
        'download': True,
    }

    # Split train into train and validation
    ds_train_full = CIFAR10(train=True, **ds_args)
    idxs = np.arange(len(ds_train_full))
    idxs_train, idxs_validation = train_test_split(idxs, test_size=args.validation_ratio, random_state=args.seed, 
                                                   shuffle=True, stratify=ds_train_full.targets)
    ds_train, ds_validation = Subset(ds_train_full, idxs_train), Subset(ds_train_full, idxs_validation)
    dss = {
        'train': TransformDataset(ds_train, transform=transform['train']),
        'train_aug': TransformDataset(ds_train, transform=transform['train_aug']),
        'validation': TransformDataset(ds_validation, transform=transform['validation']),
        'test': CIFAR10(train=False, **ds_args, transform=transform['test'])
    }
    return dss

def get_dataloaders(args):

    # Datasets
    dss = get_datasets(args)

    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dls = {
        'train': DataLoader(dss['train'], shuffle=False, **dl_args),
        'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        'test': DataLoader(dss['test'], shuffle=False, **dl_args),
    }
    return dls
