#!/usr/bin/env python3
""" train_models.py
Udacity DSDN - Breno Silva
train_models.py make models for train.py
"""

import torch

from torch import nn
from torchvision import models
from collections import OrderedDict

def get_model(arch, hidden_size, output_size, gpu):
    """
    Create and return dataloaders for training
    """

    model = models.__dict__[arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    if type(model.classifier) == nn.Sequential:
        input_size = model.classifier[0].in_features
    else:
        input_size = model.classifier.in_features
    
    if hidden_size == []:
        hidden_size = [
            (input_size // 8),
            (input_size // 32)
        ]

    ordered = OrderedDict()
    hidden_size.insert(0, input_size)

    for i in range(len(hidden_size) - 1):
        ordered['fc' + str(i + 1)] = nn.Linear(hidden_size[i], hidden_size[i+1])
        ordered['relu' + str(i + 1)] = nn.ReLU()
        ordered['dropout' + str(i + 1)] = nn.Dropout(p=0.2)

    ordered['output'] = nn.Linear(hidden_size[i + 1], output_size)
    ordered['softmax'] = nn.LogSoftmax(dim=1)
    
    classifier = nn.Sequential(ordered)
    
    model.classifier = classifier
    model.zero_grad()

    if gpu:
        model.to("cuda")
    
    return model

def train_model(model, training, validation, criterion, optimizer, epochs, gpu):
    """
    Trains the model
    """

    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in training:
            if gpu:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else: 
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation:
                    if gpu:
                        inputs, labels = inputs.to("cuda"), labels.to("cuda")

                    logps = model.forward(inputs)
                    valid_loss += criterion(logps, labels).item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs} ["
                f"train_loss: {running_loss:.3f}, "
                f"val_loss: {valid_loss/len(validation):.3f}, "
                f"val_acc: {accuracy/len(validation):.3f}]")
            model.train()