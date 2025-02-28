import argparse
import boto3.utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from PokeData import PokeData
from PokemonClassifier import PokemonClassifier
from Transform import PokeTransform
import constants

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import boto3

import os
import stat

from s3torchconnector import S3MapDataset, S3Checkpoint
import torch.distributed.checkpoint as DCP
from s3torchconnector.dcp import S3StorageWriter

def train(model, device, train_dataloader, val_dataloader, criterion, optimizer, num_epoch):
    # Loss function
    train_losses, val_losses = [], []

    for epoch in range(num_epoch):
        # Set model to train
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)

        print(f"Training Progress: Epoch {epoch} - Train loss: {train_loss} - Validation loss: {val_loss}")

def model_to_s3(model):
    model_name = 'test-model.pth'
    CHECKPOINT_URI = constants.OUTPUT_BUCKET + '/' + model_name
    # Conenct to client
    s3_storage_writer = S3StorageWriter(region=constants.REGION, path=CHECKPOINT_URI)
    DCP.save(
        state_dict=model.state_dict(),
        storage_writer=s3_storage_writer,
    )
    print(f"Model saved to {CHECKPOINT_URI}")

def get_data():
    # Create Dataset

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    DATASET_URI= constants.TRAIN_BUCKET # "s3://<BUCKET>/<PREFIX>"
    full_dataset = S3MapDataset.from_prefix(DATASET_URI, region=constants.REGION, transform=PokeTransform(transform))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    # dist.get_world_size()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    
    for images, labels in train_dataloader:
        # print(images)
        # print(labels)
        break

    return train_dataloader, val_dataloader, images, labels

def main():
    # Create and load data
    train_dataloader, val_dataloader, images, labels = get_data()

    # Create model
    model = PokemonClassifier(num_classes=149)
    example_out = model(images)

    # print(model(images).shape) # [batch_size, num_classes]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device=device)

    # Train model
    criterion = nn.CrossEntropyLoss()
    criterion(example_out, labels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,                                                                                                         
        criterion=criterion,
        optimizer=optimizer,
        num_epoch=5
    )

    path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/test-model.pth'
    torch.save(model.state_dict(), path)
    # Save model to S3
    # model_to_s3(model=model)

if __name__ == '__main__':
    main()