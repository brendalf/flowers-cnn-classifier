#!/usr/bin/env python3
""" predict.py
Udacity DSDN - Breno Silva
Deep Learning Networks - Part two of the final project
predict.py receives an image an predict using the model classification
"""

__author__ = "Breno Silva <brenophp@gmail.com>"
__version__ = "1.0.0"
__license__ = "MIT"


import os
import json
import torch
import warnings
import numpy as np

from PIL import Image
from torchvision import models

from args import get_test_args

def main():
    """
        Image Classification Prediction
    """
    
    cli_args = get_test_args(__author__, __version__)

    # Variables
    image_path = cli_args.input
    checkpoint = cli_args.checkpoint
    top_k = cli_args.top_k
    categories_names = cli_args.categories_names
    gpu = cli_args.gpu

    # If the user wants the gpu mode, check if cuda is available
    if (gpu == True) and (torch.cuda.is_available() == False):
        print("GPU mode is not available, using CPU...")
        gpu = False

    # Check the categories file
    if not os.path.isfile(categories_names):
        print(f'Categories file {categories_names} was not found.')
        exit(1)

    # Load categories
    with open(categories_names, 'r') as f:
        cat_to_name = json.load(f)

    # load model
    model = load_checkpoint(checkpoint, gpu)

    top_ps, top_classes = predict(image_path, model, top_k)

    label = top_classes[0]
    prob = top_ps[0]

    print(f'Parameters\n---------------------------------')

    print(f'Image  : {image_path}')
    print(f'Model  : {checkpoint}')
    print(f'GPU    : {gpu}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_ps)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_ps[i]*100:.2f}%")

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image.float())
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk)
        
    idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = list()
    
    for cl in top_class.numpy()[0]:
        classes.append(idx[cl])
        
    return top_p.numpy()[0], classes

def load_checkpoint(filepath, gpu):
    """
    Loads model checkpoint saved by train.py
    """
    model_state = torch.load(filepath, map_location=lambda storage, loc: storage)

    model = models.__dict__[model_state['arch']](pretrained=True)
    
    if gpu:
        model = model.to("cuda")

    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    from math import floor
    
    image = Image.open(image).convert("RGB")
    
    # Resize with aspect ratio
    width, height = image.size
    size = 256
    ratio = float(width)/float(height)
    if width > height:
        new_height = ratio * size
        image = image.resize((size, int(floor(new_height))), Image.ANTIALIAS)
    else:
        new_width = ratio * size
        image = image.resize((int(floor(new_width)), size), Image.ANTIALIAS)
    
    # Center crop
    width, height = image.size
    size = 224
    
    image = image.crop((
            (width - size) / 2,  # left
            (height - size) / 2, # top
            (width + size) / 2,  # right
            (height + size) / 2  # bottom
        ))
    
    image = np.array(image) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)


if __name__ == '__main__':
    # some models return deprecation warnings
    # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

"""
 Ensures main() only gets executed if the script is
 executed directly and not as an include.
"""