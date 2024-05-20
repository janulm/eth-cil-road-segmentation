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
DATASET CLASS FOR Custom Decoder SAM


'''

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, image, mask):
        # Convert to tensor and set dtype
        image = TF.to_tensor(image).float()
        mask = TF.to_tensor(mask).float()

        # Apply random horizontal flip to both image and mask
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Apply random vertical flip to both image and mask
        if torch.rand(1) < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Resize both image and mask
        image = TF.resize(image, (1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        mask = TF.resize(mask, (1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, mask


class sam_dataset(Dataset):
    def __init__(self, satellite_images, street_masks):
        self.satellite_images = satellite_images
        self.street_masks = street_masks
        assert len(self.satellite_images) == len(self.street_masks), "Number of images and masks should be the same"
        self.transform = CustomTransform()
        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx]
        mask = self.street_masks[idx] / 255  # normalize mask

        image, mask = self.transform(image, mask)
                
        return image, mask
