## written by Jannek Ulm on 11.7.24 for CIL ETHZ Project
#
#
#
#
#
#
#
#
#
#



import torch
import numpy as np
import random


# code was inspired by the following sources: https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
from utils.image_loading import * 
from utils.torch_device import *
from custom_datasets import Sat_Mask_Dataset, Sat_Only_Image_Dataset



def get_data_orig_and_custom(len_custom=None):
    # cutoff for the length of the custom dataset: 
    ###########
    original_data = {}
    original_data["images"] = load_training_images()
    original_data["masks"] = load_groundtruth_images()

    city_names = ["boston","nyc","zurich"]
    custom_data = {"images":[],"masks":[]} # stores images and gt masks

    for name in city_names:
        custom_data["images"].extend(load_training_images(name))
        custom_data["masks"].extend(load_groundtruth_images(name))
    
    if len_custom:
        max_len = min(len_custom, len(custom_data["images"]))
    else:
        max_len = len(custom_data["images"])
    custom_data["images"] = custom_data["images"][0:max_len]
    custom_data["masks"] = custom_data["masks"][0:max_len]

    print("the raw custom dataset contains",len(custom_data["images"]),"images")

    print("custom ds: (min,mean,max) street ratio",get_street_ratio_mmm(custom_data["masks"]))
    print("orig ds: (min,mean,max) street ratio",get_street_ratio_mmm(original_data["masks"]))

    # create a dataset
    custom_data_set = Sat_Mask_Dataset(custom_data["images"], custom_data["masks"],min_street_ratio=0.03,max_street_ratio=1.0)
    original_data_set = Sat_Mask_Dataset(original_data["images"],original_data["masks"])
    print("after cleanup, the dataset now contains",len(custom_data_set),"images")


    # submission kaggle dataset

    kaggle_submission_images = load_test_images()
    submission_data_set = Sat_Only_Image_Dataset(kaggle_submission_images)

    return original_data_set,custom_data_set,submission_data_set



if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = get_torch_device(allow_mps=True)
    print("using device:",device)

    # get the data sets
    original_data_set,custom_data_set,submission_data_set = get_data_orig_and_custom()

    # create a dataloader