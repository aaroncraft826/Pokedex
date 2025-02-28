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

from PokemonClassifier import PokemonClassifier
from Transform import PokeTransform
import constants

import boto3

import os
import sys

from s3torchconnector import S3MapDataset
import torch.distributed.checkpoint as DCP
from s3torchconnector.dcp import S3StorageWriter

class PokeTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer
        ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.epochs_run = 0

    def save_snapshot(self, epoch):
        print(f"GPU[{self.global_rank}] - Saving snapshot at {epoch}")
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model_state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        self.model_to_s3(is_snapshot=True)
        print(f"GPU[{self.global_rank}] - Save training snapshot at epoch {self.epochs_run}")

    def load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"GPU[{self.global_rank}] - Resume training at epoch {self.epochs_run}")

    def train(self, num_epoch):
        print(f"GPU[{self.global_rank}] - Begin training")

        # Loss function
        train_losses, val_losses = [], []
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epoch):
            # Set model to train
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_data:
                images, labels = images.to(self.local_rank), labels.to(self.local_rank)
                self.optimizer.zero_grad()
                outputs = self.model.module(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / len(self.train_data.dataset)
            train_losses.append(train_loss)

            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for images, labels in self.val_data:
                    images, labels = images.to(self.local_rank), labels.to(self.local_rank)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
            val_loss = running_loss / len(self.val_data.dataset)
            val_losses.append(val_loss)

            print(f"GPU[{self.global_rank}] - Training Progress: Epoch {epoch} - Train loss: {train_loss} - Validation loss: {val_loss}")

    def model_to_s3(self, is_snapshot: bool = False):
        # Create model/snapshot URI
        model_name = ""
        if is_snapshot:
            model_name = f'test-snapshot-{self.global_rank}.pth'
        else:
            print(f"GPU[{self.global_rank}] - Saving Model")
            model_name = f'test-model-{model_name}.pth'
        CHECKPOINT_URI = constants.OUTPUT_BUCKET + '/' + model_name

        # Conenct to client
        s3_storage_writer = S3StorageWriter(region=constants.REGION, path=CHECKPOINT_URI)
        DCP.save(
            state_dict=self.model_state_dict(),
            storage_writer=s3_storage_writer,
        )
        print(f"GPU[{self.global_rank}] - Model saved to {CHECKPOINT_URI}")
        
    def model_state_dict(self):
        return self.model.module.state_dict()

# rank: Unique identifier for process
# world_size: Total number of processes
def ddp_setup():
    init_process_group(backend="nccl")


def load_train_objs():
    # Create Dataset

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    DATASET_URI= constants.TRAIN_BUCKET # "s3://<BUCKET>/<PREFIX>"
    full_dataset = S3MapDataset.from_prefix(DATASET_URI, region=constants.REGION, transform=PokeTransform(transform))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")

    for images, labels in train_dataloader:
        # print(images)
        # print(labels)
        break

    return train_dataloader, val_dataloader, images, labels

def main(total_epochs: int):
    # Setup process group
    ddp_setup()

    # Create and load data
    train_data, val_data, images, labels = load_train_objs()

    # Create model
    model = PokemonClassifier(num_classes=149)
    example_out = model(images)

    # Train model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = PokeTrainer(model, train_data, val_data, optimizer)
    trainer.train(total_epochs)

    # Save model
    trainer.model_to_s3()
    # path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/test-model.pth'
    # torch.save(trainer.model_state_dict(), path)

    # Clean process group
    destroy_process_group()

if __name__ == '__main__':
    total_epochs = int(sys.argv[1])
    world_size = torch.cuda.device_count()
    print(f"Total Epochs: {total_epochs}")
    print(f"Process Count: {world_size}")
    main(total_epochs)