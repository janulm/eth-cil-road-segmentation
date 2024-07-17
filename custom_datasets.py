from torch.utils.data import Dataset
import torch



'''

Dataset Class that transforms all the images to the format required by sam preprocess

'''

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
#from albumentations import Compose, HorizontalFlip, VerticalFlip, Resize
import numpy as np

class CustomTransform:
    def __init__(self,img_size=416,do_random=False,handle_mask=True):
        self.img_size = img_size
        self.do_random = do_random
        self.handle_mask = handle_mask

    def __call__(self, image, mask):
        # Convert to tensor and set dtype
        image = TF.to_tensor(image).float()
        if self.handle_mask:
            mask = TF.to_tensor(mask).float()

        # Apply random horizontal flip to both image and mask
        if self.do_random and torch.rand(1) < 0.5:
            image = TF.hflip(image)
            if self.handle_mask: 
                mask = TF.hflip(mask)

        # Apply random vertical flip to both image and mask
        if self.do_random and torch.rand(1) < 0.5:
            image = TF.vflip(image)
            if self.handle_mask: 
                mask = TF.vflip(mask)
        
        # Resize both image and mask
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        if self.handle_mask:
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, mask

# class CustomTransformUPP:
#     def __init__(self, img_size=224, do_random=False, handle_mask=True):
#         self.img_size = img_size
#         self.do_random = do_random
#         self.handle_mask = handle_mask
#         self.transform = Compose([
#             Resize(self.img_size, self.img_size),
#             HorizontalFlip(p=0.5 if self.do_random else 0.0),
#             VerticalFlip(p=0.5 if self.do_random else 0.0),
#         ])

#     def __call__(self, image, mask):
#         # Convert to numpy array if not already
#         if not isinstance(image, np.ndarray):
#             image = np.array(image)
#         if not isinstance(mask, np.ndarray):
#             mask = np.array(mask)

#         # Apply transformations
#         augmented = self.transform(image=image, mask=mask)
#         image = augmented['image']
#         mask = augmented['mask']

#         # Convert to tensor
#         image = TF.to_tensor(image).float()
#         if self.handle_mask:
#             mask = TF.to_tensor(mask).float()

#         return image, mask

class Sat_Mask_Dataset(Dataset):
    def __init__(self, satellite_images, street_masks,min_street_ratio=0.0,max_street_ratio=1.0,img_size=1024,do_random=True):
        # do random: decides if random flips and modifications to images are done or not
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
        self.transform = CustomTransform(img_size=img_size,do_random=do_random)
        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx] # scale 
        mask = self.street_masks[idx] / 255  # normalize mask

        image, mask = self.transform(image, mask)
        image = image * 255.
        return image, mask
    

class Sat_Mask_Dataset_UPP_preprocessed(Dataset):
    def __init__(self, satellite_images, street_masks, min_street_ratio=0.0, max_street_ratio=1.0, img_size=416, do_random=True, upp_preprocess=None):
        assert len(satellite_images) == len(street_masks), "Number of images and masks should be the same"
        self.satellite_images = []
        self.street_masks = []
        count_discard = 0
        for i in range(len(satellite_images)):
            ratio = street_masks[i].mean() / 255.
            if min_street_ratio <= ratio <= max_street_ratio:
                self.satellite_images.append(satellite_images[i])
                self.street_masks.append(street_masks[i])
            else:
                count_discard += 1
        print("Initialized dataset, checked for min,max street ratio. Discarded %:", count_discard/len(street_masks), " num discarded:", count_discard)
        assert len(self.satellite_images) == len(self.street_masks), "Number of images and masks should be the same"
        self.transform = CustomTransform(img_size=img_size, do_random=do_random)
        self.upp_preprocess = upp_preprocess

    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx]
        mask = self.street_masks[idx] / 255

        image, mask = self.transform(image, mask)
        image = image * 255.
        preprocessed = self.upp_preprocess(image=image.numpy().transpose(1, 2, 0))
        image = preprocessed["image"].type(torch.float32)
        mask = mask.type(torch.float32)[0].unsqueeze(0)
        return image, mask


# use this class for just for the kaggle test images, since there are no masks available for this dataset: 
class Sat_Only_Image_Dataset(Dataset):
    def __init__(self, satellite_images,img_size=1024):
        
        self.satellite_images = satellite_images
        self.transform = CustomTransform(img_size=img_size,do_random=False,handle_mask=False)
        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx] # scale 

        image, mask = self.transform(image, None)
        image = image * 255. 
        return image


# use this class for just for the kaggle test images, since there are no masks available for this dataset: 
class Sat_Only_Image_UPP_preprocessed(Dataset):
    def __init__(self, satellite_images, img_size=416,  upp_preprocess=None):
        self.satellite_images = satellite_images
        self.transform = CustomTransform(img_size=img_size, do_random=False, handle_mask=False)
        self.upp_preprocess = upp_preprocess
        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        image = self.satellite_images[idx] # scale 

        image, mask = self.transform(image, None)
        image = image * 255.
        preprocessed = self.upp_preprocess(image=image.numpy().transpose(1, 2, 0))
        image = preprocessed["image"].type(torch.float32)
        return image