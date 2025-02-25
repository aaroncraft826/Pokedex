from typing import Tuple, List, Dict

from torchvision.datasets import ImageFolder

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

def classes_to_idx() -> dict:
    return load_json_as_dict("class_indexes.json")


class CustomImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        classes = list(classes_to_idx().keys())
        class_to_idx = classes_to_idx()
        return classes, class_to_idx