#!/usr/bin/env python3
""" train.py
Udacity DSDN - Breno Silva
Deep Learning Networks - Part two of the final project
train.py train a new network on a specified data set
"""

__author__ = "Breno Silva <brenophp@gmail.com>"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torch import optim

from train_args import get_args
from train_models import get_model, train_model
from train_dataloaders import get_dataloaders

def main():
    """
        Train an image classification network
    """

    archs = [
        'vgg11',
        'vgg13',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'densenet161',
        'densenet201'
    ]

    cli_args = get_args(__author__, __version__, archs)

    # Variables
    data_dir = cli_args.data_dir
    save_dir = cli_args.save_dir
    categories_json = cli_args.categories_json
    arch = cli_args.arch
    learning_rate = cli_args.learning_rate
    hidden_size = cli_args.hidden_size
    epochs = cli_args.epochs
    gpu = cli_args.gpu
    norm_means = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    image_size = 256
    image_size_crop = 224
    batch_size = 64

    # Check the data directory
    if not os.path.isdir(data_dir):
        print(f'Data directory {data_dir} was not found.')
        exit(1)

    # Load the data
    dataloaders = get_dataloaders(data_dir, norm_means, norm_std, image_size, image_size_crop)

    # Check the categories file
    if not os.path.isfile(categories_json):
        print(f'Categories file {categories_json} was not found.')
        exit(1)

    # Load categories
    with open(categories_json, 'r') as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name)

    # Check the arch choosen
    if not arch in archs:
        print("Not supported architecture.")
        exit(1)

    # If the user wants the gpu mode, check if cuda is available
    if (gpu == True) and (torch.cuda.is_available() == False):
        print("GPU mode is not available, using CPU...")
        gpu = False
    
    # Make the model
    model = get_model(arch, hidden_size, output_size, gpu)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloaders['training'], dataloaders['validation'], criterion, optimizer, epochs, gpu)

    # Check and create the save directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
"""
 Ensures main() only gets called if the script is executed directly and not as an include. 
 Prevent stacktrace on Ctrl-C.
"""