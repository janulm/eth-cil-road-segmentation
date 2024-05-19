# ETHZ Computational Intelligence Lab 2024 Road Segmentation Project


## TODO:

### SAM

- https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=jtPYpirbK3Wi check out this, this is how we can use the sam model, maybe also with larger batch size... ?
- there could be a big speedup by modifying the the dataset, and preprocessing the images in such a way, as the SamPredictor does when setting an image each time. this cost can be then done once and is not required each time the image is computed. Even better would be assuming that the image encoder is never finetuned, to simply procompute the image embeddings and store them in a file. 
- it works for sure with batch size > 1 when using the approach from Huggingace Transformers as showed here: https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
  


## Interesting papers

1. https://arxiv.org/pdf/2403.16051 
2. 

## Datasets:

1. the one given from class
2. self generated and downloaded from google api? 




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

