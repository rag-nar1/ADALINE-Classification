# convert images to a 1d array with the pixel values

import numpy as np
from PIL import Image
import os

def hash(image):
    hashed = np.array([])
    for i in range(0, len(image), 3):
        hashed = np.append(hashed, (image[i] * 2 + image[i + 1] * 4 + image[i + 2] * 8))
    return hashed

class TrainingData:

    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.imagedir = self.dir_path + '/images/Training/'
        self.appledir = self.imagedir + 'Apple1'
        self.bananadir = self.imagedir + 'Banana2'


    def image_to_array(self ,image):
        return np.array(image).flatten()


    def load_images_from_folder(self , folder):
        images = []
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename))
            img = img.resize((100, 100))
            if img is not None:
                images.append(self.image_to_array(img))
        return images

    def load_images(self):
        apples = self.load_images_from_folder(self.appledir)
        bananas = self.load_images_from_folder(self.bananadir)
        # print("finished loading images")
        return apples, bananas

    def main(self):
        apples, bananas = self.load_images()
        # convert numpy array to list
        for i in range(len(apples)):
            apples[i] =  hash(apples[i].tolist())

        for i in range(len(bananas)):
            bananas[i] = hash(bananas[i].tolist())
        return apples, bananas
