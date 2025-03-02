import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from PokeData import PokeData
from PokemonClassifier import PokemonClassifier

import matplotlib.pyplot as plt

import os
import stat

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

def main():
    # Create Dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    data_dir = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/dataset'
    full_dataset = PokeData(data_dir, transform)
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2, 0.0])

    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    #print(test_dataset.classIds)
    print(full_dataset.classIds)

    # image, label = dataset[6000]
    # print(image, label)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for images, labels in train_dataloader:
        print(images)
        print(labels)
        break

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

    path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/model1.pth'
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    main()