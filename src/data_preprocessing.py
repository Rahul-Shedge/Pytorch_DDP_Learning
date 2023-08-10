import requests
import zipfile
import torch
import yaml
from torch import nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils.utils import read_yaml
from src.data_transforms import torchvision_transform_train,torchvision_transform_test
from torch.utils.data.distributed import DistributedSampler



def create_dataloaders(
    train_dir:str,
    test_dir:str,
    # transforms:transforms.Compose,
    batch_size:int,
    # num_workers:int
):

    """
    Args:



    Returns:

    """
    # train_transform = transforms.Compose([
    #     # Resize the images to 64x64
    #     transforms.Resize(size=(64, 64)),
    #     # Flip the images randomly on the horizontal
    #     transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    #     # Turn the image into a torch.Tensor
    #     transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    # ])

    train_transform = torchvision_transform_train()
    test_transform = torchvision_transform_test()

    # test_transform = transforms.Compose([
    #     transforms.Resize(size=(64,64)),
    #     transforms.ToTensor()
    # ])



    train_data = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )



    test_data = datasets.ImageFolder(
        test_dir,
        transform= test_transform
    )

    # Get class names
    class_names = train_data.classes


    TrainDataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False, # Sampler will manage shuffling
        # num_workers = num_workers,
        pin_memory=True,
        sampler = DistributedSampler(train_data)

    )



    TestDataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        # num_workers = num_workers,
        pin_memory=True,
        sampler = DistributedSampler(test_data)

    )

    return TrainDataloader,TestDataloader, class_names










