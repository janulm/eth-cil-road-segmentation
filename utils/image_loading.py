import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# Utility function that takes a set of png images and converts them to arrays of RGB values
def load_images(image_folder):
    folder_path = os.path.join(os.path.dirname(__file__), image_folder)
    image_arrays = []

    images_names = os.listdir(folder_path)
    # sort them
    images_names.sort()

    for file_name in images_names:
        if file_name.endswith('.png'):
            path = os.path.join(folder_path, file_name)
            im_frame = Image.open(path)
            im_frame = im_frame.convert('RGB')
            np_frame = np.array(im_frame)
            image_arrays.append(np_frame)
    
    return image_arrays

# Load the training images as arrays of RGB values
def load_training_images(city_name='original'):
    if city_name == "original":
        image_folder = '../data/original/training/images'
    else:
        image_folder = "../data/"+city_name+"/images"
    return load_images(image_folder)

# Load the groudtruth images as arrays of RGB values
def load_groundtruth_images(city_name='original'):
    if city_name == "original":
        image_folder = '../data/original/training/groundtruth'
        return load_images(image_folder)
    else:
        image_folder = "../data/"+city_name+"/groundtruth"
        masks = load_images(image_folder)

        # since the masks of the not original dataset contains more unique values than 0 and 255, for grey values,
        # cast them to 0 and 255. 
        mean = 126
        for mask in masks:
            mask[mask < mean] = 0
            mask[mask >= mean] = 255
        return masks

# Load the test images as arrays of RGB values
def load_test_images():
    image_folder = '../data/original/test/images'
    return load_images(image_folder)

# Print an image based on array of RGB values
def show_image(rgb_array):
    plt.imshow(rgb_array)
    plt.show()

def get_street_ratio_mmm(mask_list):
    # expects a list of gt masks:
    # returns ratio of how much of the mask is street
    # returns min, mean and max ratio. 
    r_min = 1.
    r_max = 0.
    r_mean = 0.
    for mask in mask_list:
        ratio = mask.mean() / 255.
        r_max = max(r_max, ratio)
        r_min = min(r_min, ratio)
        r_mean += ratio
    r_mean = r_mean / len(mask_list)
    return r_min, r_mean, r_max

def get_street_ratio_distr(mask_list):
    # expects a list of gt masks:
    # returns ratio of how much of the mask is street
    # returns min, mean and max ratio. 
    ratios = []
    for mask in mask_list:
        ratio = mask.mean() / 255.
        ratios.append(ratio)
    return ratios