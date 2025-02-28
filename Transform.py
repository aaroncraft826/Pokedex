from s3torchconnector import S3Reader
import torch
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import json

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

class PokeTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, object: S3Reader) -> torch.Tensor:
        image_pil = None
        content = object.read()
        try:
            image_pil = Image.open(BytesIO(content))
        except UnidentifiedImageError:
            print(f"IMAGE FAILURE ON OBJECT {object.key}")
            # return (torch.empty((3, 256, 256), dtype=torch.int64), 0)
        image_tensor = self.transform(image_pil)
        mon_to_idx = load_json_as_dict('class_indexes.json')
        label = mon_to_idx[object.key.split('/')[0]]

        return (image_tensor, label)