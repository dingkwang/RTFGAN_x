import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import os
import glob
#import cv2
#import scipy
#import skimage
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import utils as tvutils

import models

class NumpyDataset(Dataset):
	"""docstring for NumpyDataset"""
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform

def check_dir(PATHS):
	for path in PATHS:
		if not os.path.exists(path):
			os.makedirs(path)
			print(path, 'created')
		else:
			print(path, 'already exists')

def np_loader(PATH):
	sample = np.load(PATH)
	return sample	

def normalize(I):
    I_norm = (I - np.amin(I))/(np.amax(I) - np.amin(I))
    return I_norm

def preplot(image):
    image = np.transpose(image, (1,2,0))
    image_color = np.zeros_like(image); 
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    out_image = np.flipud(np.clip(image_color, 0,1))
    return out_image[60:,62:-38,:]

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(12,8))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator_MSE+Adv', alpha=0.5)
    plt.plot(losses.T[2], label='Generators_MSE', alpha=0.5)
    plt.plot(losses.T[3], label='Generator_Adv', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.close() #plt.show()

def plot_losses(losses, name='figure', RESULTS_DIR='./', method='GAN'):
    if method == 'psf_test':
        rows, cols = (1, 3)
        fig, ax = plt.subplots(figsize=(20, 10))
        losses = np.asarray(losses)
        #print(losses.T)
        plt.subplot(rows, cols, 1); plt.plot(losses.T[0], label='Generator_MSE+Adv', alpha=0.5)
        plt.title('Generator Loss'); plt.ylabel('$L_{gen}$')
        
        plt.subplot(rows, cols, 2); plt.plot(losses.T[1], label='Generators_MSE', alpha=0.5)
        plt.title("Generator_MSE"); plt.ylabel('$Loss$')
        
        #plt.subplot(rows, cols, 3); plt.plot(losses.T[2], label='Generator_PSF', alpha=0.5)
        #plt.title("Generator_PSF"); plt.ylabel('$Loss$')
    else:
        if method == 'WGAN':
            rows, cols = (3, 2)
        else:
            rows, cols = (2, 2)
        fig, ax = plt.subplots(figsize=(20, 10))
        losses = np.asarray(losses)
        #print(losses.T)
        plt.subplot(rows, cols, 1)
        plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
        plt.title('Discriminator Loss')
        plt.ylabel('$L_{disc}$')
        plt.subplot(rows, cols, 2)
        plt.plot(losses.T[2], label='D_real', alpha=0.5)
        plt.plot(losses.T[3], label='D_fake', alpha=0.5)
        plt.title('Discriminator Output')
        plt.ylabel('$D(I)$')
        plt.legend()
        plt.subplot(rows, cols, 3)
        plt.plot(losses.T[1], label='Generator_MSE+Adv', alpha=0.5)
        plt.title('Generator Loss')
        plt.ylabel('$L_{gen}$')
        plt.subplot(rows, cols, 4)
        plt.plot(losses.T[4], label='Generators_MSE', alpha=0.5)
        plt.plot(losses.T[5], label='Generator_Adv', alpha=0.5)
        plt.title("Training Losses")
        plt.ylabel('$Loss$')
        plt.legend()
        if method == 'WGAN':
            plt.subplot(rows, cols, 5)
            plt.plot(losses.T[6], label='Wasserstein_Dist', alpha=0.5)
            plt.title("Wasserstein Distance")
            plt.ylabel('$Wass_Dist$')
            plt.subplot(rows, cols, 6)
            plt.plot(losses.T[7], label='Gradient_Penalty', alpha=0.5)
            plt.title("Gradient Penalty")
            plt.ylabel('$Grad_Penalty$')
    plt.savefig(RESULTS_DIR + name + '.png')
    plt.close()  # plt.show()

# ############################################
#     LOAD DATASET
# ############################################
def load_data(DATA_DIR, subset, batch_size, num_workers):
    transform = transforms.Compose([transforms.ToTensor()])
    # Load Train/Val/Test Dataset
    data = datasets.DatasetFolder(root=DATA_DIR + subset + '/', loader=np_loader, transform=transform,
                                  extensions=('.npy'))
    # Split Input and GT Training images
    diff_data = torch.utils.data.Subset(data, list(range(len(data) // 2)))
    gt_data = torch.utils.data.Subset(data,list(range(len(data) // 2, len(data))))
    # Prepare Training data loaders
    diff_loader = DataLoader(diff_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    gt_loader = DataLoader(gt_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    print(subset, 'Classes:', data.classes)
    print('# of', subset, 'INPUT images:', len(diff_data)); 
    print('# of', subset,'GT images:', len(gt_data))
    
    return diff_loader, gt_loader


# ############################################
#     GRADIENT PENALTY
# ############################################
def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device, LAMBDA=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 1, 480, 640)
    alpha = alpha.to(device) #if (device.type == 'cuda') else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if (device.type == 'cuda') else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
