############## INFRASTRUCTURE ##############
# code was taken from an older project to make logging easy: https://github.com/janulm/deep-learning-advanced-initialization/blob/main/infrastructure.py

# This file contains all the imports and definitions of the code that is reused many times in the project.
from typing import List
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch  
import torchvision 

from torch.optim import SGD, Adam, lr_scheduler
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset


def get_empty_tracked_params():
    return {
        'train_loss': [],
        'val_loss': [],
        'lr_list': [],
        'train_acc_top1': [],
        'train_acc_top5': [],
        'val_acc_top1': [],
        'val_acc_top5': []
    }

def plot_training(tracked_params, name, plot=True, save=False, save_path="./plots/"):
    # Plot the training curves
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    desc = f"Model {name} was trained for {tracked_params['epochs']} epochs, with a weight decay of {tracked_params['weight_decay']}, a learning rate of {tracked_params['lr']} and a momentum of {tracked_params['momentum']} and reduce factor of {tracked_params['reduce_factor']}."
   
    # plot the training loss together with the learning rate
    # compute the x-axis for the train loss, sampled every epoch
    # set title for the plot    
    x1 = np.arange(0, len(tracked_params['train_loss']), 1)
    axs[0].plot(x1,tracked_params['train_loss'], label='train_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(desc)

    # add the val_loss to the plot
    # compute the x-axis for the val loss, sampled every "tracking_freq" epoch
    stop = len(tracked_params['val_loss'])*tracked_params['tracking_freq']
    x2 = np.arange(0, stop, tracked_params['tracking_freq'])
    axs[0].plot(x2, tracked_params['val_loss'], label='val_loss')
    axs[0].legend(loc=1)
    
    # add the learning rate to the plot
    ax2 = axs[0].twinx()
    ax2.plot(tracked_params['lr_list'], label='learning_rate', color='red')
    ax2.set_ylabel('Learning Rate') 
    
    # plot the training accuracy
    axs[1].plot(x2,tracked_params['train_acc_top1'], label='train_acc')
    # plot the validation accuracy
    axs[1].plot(x2,tracked_params['val_acc_top1'], label='val_acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    axs[1].legend()
    # add some spacing between plots 
    if save: 
        fig.savefig(save_path+'.png')
    if plot:    
        plt.show()
    
    # free up memory
    fig.clear()
    plt.close(fig)


def plot_trainings(tracked_params1, tracked_params2, name1, name2):
    # Plot the training curves for two tracked_params on the same plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot train_loss and val_loss for model 1
    x1 = np.arange(0, len(tracked_params1['train_loss']), 1)
    axs[0].plot(x1, tracked_params1['train_loss'], label=f'train_loss - {name1}')
    axs[0].plot(x1, tracked_params1['val_loss'], label=f'val_loss - {name1}')
    
    # Plot train_loss and val_loss for model 2
    x2 = np.arange(0, len(tracked_params2['train_loss']), 1)
    axs[0].plot(x2, tracked_params2['train_loss'], label=f'train_loss - {name2}')
    axs[0].plot(x2, tracked_params2['val_loss'], label=f'val_loss - {name2}')
    
    print('plotting train loss: 1:',tracked_params1["train_loss"],'2:',tracked_params2["train_loss"])
    print('plotting val loss: 1:',tracked_params1["val_loss"],'2:',tracked_params2["val_loss"])

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss and Validation Loss')
    axs[0].legend()
    
    # Plot for train_acc and val_acc
    axs[1].plot(x1, tracked_params1['train_acc_top1'], label=f'train_acc - {name1}')
    axs[1].plot(x1, tracked_params1['val_acc_top1'], label=f'val_acc - {name1}')
    axs[1].plot(x2, tracked_params2['train_acc_top1'], label=f'train_acc - {name2}')
    axs[1].plot(x2, tracked_params2['val_acc_top1'], label=f'val_acc - {name2}')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training Accuracy and Validation Accuracy')
    axs[1].legend()
    
    plt.suptitle(f"Comparison of Training Curves for {name1} and {name2}")
    
    # save the plot
    plt.savefig(f'./plots/{name1}_{name2}.png')
    #plt.show()
    # free up memory
    fig.clear()
    plt.close(fig)

def plot_trainings_mean_min_max(tracked_params_dict, display_train_acc, display_only_mean, save, save_path, display, display_max_instead_of_mean=False): 
    # dict is of the form:
    # {"model_name": tracked_params(mean,min,,max), ...}
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # Plot for train_loss and val_loss for each model
    colors = ["blue","orange","green","red","purple","brown","yellow","gray","olive","cyan","magenta","pink","black","darkblue","darkorange","darkgreen","darkred","darkpurple","darkbrown","darkpink","darkgray","darkolive","darkcyan","darkmagenta","darkyellow","darkblack"]
    # reverse colors array
    colors = colors[::-1]
    for model_name, (p_mean,p_min,p_max) in tracked_params_dict.items():
        color = colors.pop()
            
        # plot the mean values: 
        x1 = np.arange(1, len(p_mean['train_loss'])+1, 1)
        
        if display_max_instead_of_mean:
            axs.plot(x1,p_max['val_acc_top1'], label=f'max val acc - {model_name}',color=color)
        else:
            axs.plot(x1,p_mean['val_acc_top1'], label=f'mean val acc - {model_name}',color=color)
        
        if display_train_acc:
            train_color = colors.pop()
            axs.plot(x1,p_mean['train_acc_top1'], label=f'mean train acc - {model_name}',color=train_color)
        
        
        # also plot min and max data: 
        if not display_only_mean:
            # plot val_acc min,max range
            axs.fill_between(x1, p_min['val_acc_top1'], p_max['val_acc_top1'], alpha=0.2,color=color)

            if display_train_acc: 
            # plot train_acc min,max range
                axs.fill_between(x1, p_min['train_acc_top1'], p_max['train_acc_top1'], alpha=0.2,color=train_color)
            
    # add legend and titles to the plot
    axs.legend()
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Validation accuracy')
        
    # save the image to disk
    if save:
        fig.savefig(save_path+'.png')

    if display:
        plt.show()
        plt.close(fig)

def list_tracked_params_to_avg(list_tracked_params, also_min_max=False):
    if not also_min_max:
        # Compute the average for each dimension in tracked_params
        avg_tracked_params = {}
        # for all keys except the individual losses and accuracies just copy the ones, since they are all the same
        for key in list_tracked_params[0].keys():
            avg_tracked_params[key] = list_tracked_params[0][key]
    
        keys = ['train_loss','val_loss','train_acc_top1','train_acc_top5','val_acc_top1','val_acc_top5',"lr"]
        for key in keys:
            avg_tracked_params[key] = np.mean([params[key] for params in list_tracked_params], axis=0)
        return avg_tracked_params
    else: 
        avg_tracked_params = {}
        min_tracked_params = {}
        max_tracked_params = {}
        for key in list_tracked_params[0].keys():
            avg_tracked_params[key] = list_tracked_params[0][key]
            min_tracked_params[key] = list_tracked_params[0][key]
            max_tracked_params[key] = list_tracked_params[0][key]
        keys = ['train_loss','val_loss','train_acc_top1','train_acc_top5','val_acc_top1','val_acc_top5',"lr"]
        for key in keys:
            avg_tracked_params[key] = np.mean([params[key] for params in list_tracked_params], axis=0)
            min_tracked_params[key] = np.min([params[key] for params in list_tracked_params], axis=0)
            max_tracked_params[key] = np.max([params[key] for params in list_tracked_params], axis=0)
        return avg_tracked_params, min_tracked_params, max_tracked_params

def plot_training_avg(list_tracked_params,name,plot=True, save=False):
    avg_tracked_params = list_tracked_params_to_avg(list_tracked_params)
    # Visualize the average tracked_params using the plot_training function
    plot_training(avg_tracked_params, name,plot,save)