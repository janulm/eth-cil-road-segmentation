from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2
import torchvision
import torchvision.transforms.v2 as transforms
import torch

# new imports
from torchvision import transforms
import torchvision.transforms.functional as TF


'''

Dataset Class that transforms all the images to the format required by sam preprocess

'''

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class CustomTransform:
    def __init__(self,img_size=1024):
        self.img_size = img_size

    def __call__(self, image, mask):
        # Convert to tensor and set dtype
        image = TF.to_tensor(image).float()
        mask = TF.to_tensor(mask).float()

        # Apply random horizontal flip to both image and mask
        #if torch.rand(1) < 0.5:
        #    image = TF.hflip(image)
        #    mask = TF.hflip(mask)

        # Apply random vertical flip to both image and mask
        #if torch.rand(1) < 0.5:
        #    image = TF.vflip(image)
        #    mask = TF.vflip(mask)
        
        # Resize both image and mask
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, mask


class Sat_Mask_Dataset(Dataset):
    def __init__(self, satellite_images, street_masks,min_street_ratio=0.0,max_street_ratio=1.0):
        # filters out all images and their corresponding mask that are outside of the range of this dataset. 
        assert len(satellite_images) == len(street_masks), "Number of images and masks should be the same"
        self.satellite_images = []
        self.street_masks = []
        count_discard = 0
        for i in range(len(satellite_images)):
            # compute ratio
            # check if in bounds, then append. 
            ratio = street_masks[i].mean() / 255.
            if (ratio >= min_street_ratio and ratio <= max_street_ratio):
                self.satellite_images.append(satellite_images[i])
                self.street_masks.append(street_masks[i])
            else: 
                count_discard += 1
        print("Initialzed dataset, checked for min,max street ratio. Discarded %:",count_discard/len(street_masks)," num discarded:",count_discard)
        assert len(self.satellite_images) == len(self.street_masks), "Number of images and masks should be the same"
        self.transform = CustomTransform(img_size=1024)
        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx] # scale 
        mask = self.street_masks[idx] / 255  # normalize mask

        image, mask = self.transform(image, mask)
        image = image * 255. 
        mask = mask[0].unsqueeze(0) 
        return image, mask
