import numpy as np
import random
from PIL import Image
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import os

train_dir = '/home/harsh/Projects/data/train/ben_afflek'


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def augment():
    num_files_desired = 10

    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }

    # find all files paths from the folder
    images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if
              os.path.isfile(os.path.join(train_dir, f))]
    print(images)
    num_generated_files = 0
    transformed_image = None
    while num_generated_files <= num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)

        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        while num_transformations <= num_transformations_to_apply:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            print(num_transformations)
            num_transformations += 1

        new_file_path = '%s/new/augmented_image_%s.jpg' % (train_dir, num_generated_files)

        # write image to the disk
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1
        print(num_generated_files)


augment()
print("Done")
