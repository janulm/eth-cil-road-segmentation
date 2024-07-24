# ETHZ - Computational Intelligence Lab 2024 - Road Segmentation Project




- [ETHZ - Computational Intelligence Lab 2024 - Road Segmentation Project](#ethz---computational-intelligence-lab-2024---road-segmentation-project)
  - [Team](#team)
  - [Project Description](#project-description)
  - [Running the code](#running-the-code)
    - [Installing the required packages](#installing-the-required-packages)
    - [Downloading SAM checkpoints](#downloading-sam-checkpoints)

## Team 
-  **Jannek Ulm** (https://github.com/janulm)
-  **Douglas Orisini-Rosenberg** 
- **Raoul van Doren** 
- **Paul Ellsiepen**

## Project Description

The project is based on the Kaggle competition "Road Segmentation" (https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2024). The dataset consists of 144 satellite images of size 400x400 pixels, with corresponding ground truth labels. The goal is to predict the road segmentation of the images. For evaluation the images are split into 16x16 patches and F1-Score is reported.

We ranked first in the overall kaggle competition.

Here is a sample prediction of our final model on the test set:
![sample_prediction](qualitative_example.png)



## Running the code 

If you want to run the code, please follow the instructions below. First one needs to install the required packages. This can be done by running the following command in the terminal:

### Installing the required packages
Note that this requires a machine with CUDA support, since some code makes use of CUDA proprieatry optimizations such as TF32 and torch.compile(). 
    
    conda env create -f env.yml
    conda activate cil



### Downloading SAM checkpoints
Furthermore, the SAM model checkpoints need to be downloaded from the following link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth



![BiSeSAM Paper](BiSeSAM.pdf)




