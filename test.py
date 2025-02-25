import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from main import PokemonClassifier
from PIL import Image
from CustomImageFolder import load_json_as_dict
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test():
    path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/model1.pth'
    model = PokemonClassifier()
    model.load_state_dict(torch.load(path, weights_only=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()

    mon_to_idx = load_json_as_dict("class_indexes.json")
    idx_to_mon = {v: k for k, v in mon_to_idx.items()}

    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/pikachu.jpg"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Pikachu")
    
    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/eevee.webp"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Eevee")
    
    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/charmander.jpg"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Charmander")
    
    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/squirtle.jpg"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Squirtle")
    
    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/bulbasaur.webp"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Bulbasaur")
    
    image_path = "/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/exampleset/oddish.jpg"
    prediction = test_image(model=model, device=device, img_path=image_path, idx_to_class=idx_to_mon)
    print(f"Predicted Pokemon: {prediction} - Answer: Oddish")

def test_image(model, device, img_path, idx_to_class):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    preprocessed_img = transform(img)
    preprocessed_img = torch.unsqueeze(preprocessed_img, 0)
    # preprocessed_img.to(device)

    with torch.no_grad():
        prediction = model(preprocessed_img)
    
    probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return idx_to_class[predicted_class]

if __name__ == "__main__":
    test()