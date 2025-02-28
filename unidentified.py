import glob
import PIL
from PIL import Image
from PIL import UnidentifiedImageError
import os
from io import BytesIO

def main():
    dir_path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/dataset256x256/'
    dirs = os.listdir(dir_path)

    for folder in dirs:
        folder_path = dir_path + folder + '/'
        files = os.listdir(folder_path)
        for item in files:
            if os.path.isfile(folder_path + item):
                with open(folder_path + item, 'rb') as file:
                    file_data = file.read()
                    try:
                        img = Image.open(BytesIO(file_data))
                    except PIL.UnidentifiedImageError:
                        print(folder_path + item)
                
if __name__ == '__main__':
    main()