import os
from PIL import Image
import constants


def resize_image(dir_path, folder_name, img_name):
    file_path = dir_path + folder_name + '/' + img_name
    out_dir = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/dataset256x256/'
    try:
        with Image.open(file_path) as img:
            img = img.resize(constants.IMAGE_SIZE)
            if os.path.isdir(out_dir + folder_name) == False:
                os.mkdir(out_dir + folder_name)
            img.save(out_dir + folder_name + '/' + img_name, 'PNG')
    except IOError as e:
        print("An exception occured '%s'" %e)

def main():
    dir_path = '/mnt/c/Users/agice/Desktop/Side_Projects/Pokedex_Data/dataset/'
    dirs = os.listdir(dir_path)

    for folder in dirs:
        folder_path = dir_path + folder + '/'
        files = os.listdir(folder_path)
        for item in files:
            if os.path.isfile(folder_path + item):
                print(folder_path + item)
                resize_image(dir_path, folder, item)

if __name__ == '__main__':
    main()