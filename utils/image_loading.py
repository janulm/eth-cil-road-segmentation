import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# Utility function that takes a set of png images and converts them to arrays of RGB values
def load_images(image_folder):
    folder_path = os.path.join(os.path.dirname(__file__), image_folder)
    image_arrays = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            path = os.path.join(folder_path, file_name)
            im_frame = Image.open(path)
            im_frame = im_frame.convert('RGB')
            np_frame = np.array(im_frame)
            image_arrays.append(np_frame)
    
    return image_arrays

# Load the training images as arrays of RGB values
def load_training_images():
    image_folder = '../data/training/images'
    return load_images(image_folder)

# Load the groudtruth images as arrays of RGB values
def load_groundtuth_images():
    image_folder = '../data/training/groundtruth'
    return load_images(image_folder)

# Load the test images as arrays of RGB values
def load_test_images():
    image_folder = '../data/test/images'
    return load_images(image_folder)

# Print an image based on array of RGB values
def show_image(rgb_array):
    plt.imshow(rgb_array)
    plt.show()