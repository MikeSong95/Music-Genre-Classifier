"""
Helper functions to sort data into correct folder structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent
from datetime import datetime
torch.manual_seed(123) # set the random seed

import os
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from load_data import load_data
from model import MusicClassifier

def get_accuracy(model, data):
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64,shuffle=False ):
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=30, lr=0.001, momentum=0.01):
    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=batch_size,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    epochs, iters, losses, train_acc, val_acc = [], [], [], [], []
    
    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        epochs.append(epoch)
        print(("Epoch {}:").format(epoch + 1))
        start_time = time.time()
        for imgs, labels in iter(train_loader):
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            n += 1
        if epoch%5==0:
            train_acc.append(get_accuracy(model, data["train"])) # compute training accuracy 
            val_acc.append(get_accuracy(model, data["val"]))  # compute validation accuracy
            print(("Train accuracy: {}, Train loss: {} \n"+"Validation accuracy: {}").format(
                    train_acc[-1],
                    losses[-1],
                    val_acc[-1]))
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    print("Final Training Accuracy: {}".format(train_acc[-1]))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
data = load_data()
model = MusicClassifier()
model.load_state_dict(torch.load("./music_classifier.pt"))
train(model, data)
torch.save(model.state_dict(), "./music_classifier.pt")