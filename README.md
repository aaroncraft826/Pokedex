# Pokedex

## Overview

This is a repository for training an image classifier for gen 1 pokemon
It contains the infrastructure to create an eks cluster via terraform, in order to run the trainer distributedly.

## Commands

local: torchrun --standalone --nproc_per_node=gpu PokeTrainer.py 5

global: torchrun --nproc_per_node=gpu --nnodes=3 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint={ip_addr:port} PokeTrainer.py 10

nvidia device plugin to access gpu resources
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml