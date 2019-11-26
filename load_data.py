import os
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

def load_data():
    # define training and test data directories
    data_dir = './data/'
    train_dir = os.path.join(data_dir, 'train/')
    val_dir = os.path.join(data_dir, 'val/')
    test_dir = os.path.join(data_dir, 'test/')
    overfit_dir = os.path.join(data_dir, 'overfit/')

    # resize all images to 224 x 224
    data_transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    val_data = datasets.ImageFolder(val_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    overfit_data = datasets.ImageFolder(overfit_dir, transform=data_transform)
    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num validation images: ', len(val_data))
    print('Num test images: ', len(test_data))
    print('Num overfit images: ', len(overfit_data))

    return {"train":train_data, "val":val_data, "test":test_data, "overfit":overfit_data}
