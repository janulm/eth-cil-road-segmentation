# ETHZ Computational Intelligence Lab 2024 Road Segmentation Project




## TODO/Ideas:

### Dataset: 

Note: For evaluation on kaggle only the path wise (16x16 label is checked)
Instead of pixle wise prediction we could try to predict for each of the 16x16 patches a binary street or not label. 
This would reduce the amount of data we have to predict by a factor of 256. And maybe makes it easier for the model to learn. 

Test approach using F1 score for evaluation or other metrics such as BCE, Dice.

We are not evaluated on accuracy but on the F1 score. (better measure for unbalanced datasets)

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









## Getting things started: 

### Installing SAM using pip on local conda env: 
    pip install git+https://github.com/facebookresearch/segment-anything.git

### Downloading a pretrained SAM model checkpoint: 
    Used default version from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


### Data: Both training and test folder are placed in the data folder.

