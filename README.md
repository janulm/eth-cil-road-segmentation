# ETHZ Computational Intelligence Lab 2024 Road Segmentation Project

Note: For evaluation on kaggle only the path wise (16x16 label is checked)
    If 

Ideas: maybe not only work on the 400x400 images but only on the 16x16 image patches, 
Think about data balancing since there are much more non-road pixels than road pixels./ patches


We are not evaluated on accuracy but on the F1 score. (better measure for unbalanced datasets)




## Getting things started: 

### Installing SAM using pip on local conda env: 
    pip install git+https://github.com/facebookresearch/segment-anything.git

### Downloading a pretrained SAM model checkpoint: 
    Used default version from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


### Data: Both training and test folder are placed in the data folder.

