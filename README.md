# Pokedex

## Overview

This is a repository for training an image classifier for gen 1 pokemon
It contains the infrastructure to create an eks cluster via terraform, in order to run the trainer distributedly.

The data used is from:
https://www.kaggle.com/datasets/thedagger/pokemon-generation-one/data
https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data

## Instructions

### Setup training infrastructure

#### Use terraform to create infrastructure
cd terraform
terraform init
terraform apply

#### Download training data
download the kaggle set and upload it to the s3 training bucket "poke-train-bucket"
https://www.kaggle.com/datasets/thedagger/pokemon-generation-one/data

#### Give nodegroup s3 access
current version doesn't give the kubernetes nodegroup
in aws, find the nodegroup's IAM role and attach s3fullaccess to its policy

### Running the program

#### Create docker image and upload to ecr repository
make docker

#### Update script and deployment/daemonset.yaml and run on cluster
make deploy

--rdzv_endpoint=10.0.1.243:12345

## Commands

local: torchrun --standalone --nproc_per_node=gpu PokeTrainer.py 5

global: torchrun --nproc_per_node=gpu --nnodes=3 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint={ip_addr:port} PokeTrainer.py 10

nvidia device plugin to access gpu resources
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml