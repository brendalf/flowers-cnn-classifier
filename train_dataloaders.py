#!/usr/bin/env python3
""" train_dataloaders.py
Udacity DSDN - Breno Silva
train_dataloaders.py define transforms e imagesets and return dataloaders for train.py
"""

import torch
from torchvision import datasets, transforms

def get_dataloaders(data_dir, norm_means, norm_std, image_size=256, image_size_crop=224, batch_size=64):
    """
    Create and return dataloaders for training
    """

    # Testing with data_dir has / in the end
    if data_dir[-1] != "/":
        data_dir += "/"

    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    # Data Transforms
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_size_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_std)
        ]), 'validation': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size_crop),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_std)
        ])
    }

    # Image Datasets
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    # Data Loaders
    return image_datasets, {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size)
    }
