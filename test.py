import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from PokemonClassifier import PokemonClassifier
from PIL import Image

import matplotlib.pyplot as plt
import json

from s3torchconnector.dcp import S3StorageReader
import constants as constants
import torch.distributed.checkpoint as DCP

def load_json_as_dict(file_path):
    try:
        with open('poke_indexes.json', 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None

def test():
    MODEL_URI = constants.OUTPUT_BUCKET + '/' + 'test-model-.pth' + '/'
    model = PokemonClassifier()
    model_state_dict = model.state_dict()
    s3_storage_reader = S3StorageReader(region=constants.REGION, path=MODEL_URI)
    DCP.load(
        state_dict=model_state_dict,
        storage_reader=s3_storage_reader
    )
    model.load_state_dict(model_state_dict)
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