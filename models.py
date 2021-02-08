import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import utils as tvutils
from torchvision.models.vgg import vgg16


def define_D(netD='basic'):
    if netD == 'basic':
        discriminator = Discriminator()
    elif netD == 'wgan':
        discriminator = Discriminator_WGAN()
    
    return discriminator

def define_G(netG='unet'):
    if netG == 'unet':
        generator = UNet((3, 256, 256))
    if netG == 'unet_double':
        generator = UNet_doubleDec((3,256,256))
    
    return generator

BN_EPS = 1e-4
class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small
        
class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2
    
        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True) #Updated from F.upsample()
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, in_shape):
        super(UNet, self).__init__()
        channels, height, width = in_shape
    
        self.down1 = StackEncoder(channels, 24, kernel_size=3) ;# 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
            
    
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, channels, kernel_size=1, bias=True)
    
    
        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        #self.center = nn.Sequential(ConvBnRelu2d(256, 256, kernel_size=3, padding=1))
    
    def forward(self, x):
        out = x; 
        down1, out = self.down1(out); 
        down2, out = self.down2(out); 
        down3, out = self.down3(out); 
        down4, out = self.down4(out); 
        down5, out = self.down5(out); 
    
        out = self.center(out)
        out = self.up5(out, down5); 
        out = self.up4(out, down4); 
        out = self.up3(out, down3); 
        out = self.up2(out, down2); 
        out = self.up1(out, down1); 
    
        out = self.classify(out); 
        out = torch.squeeze(out, dim=1); 
        return out
    
class UNet_small(nn.Module):
    def __init__(self, in_shape):
        super(UNet_small, self).__init__()
        channels, height, width = in_shape
    
        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 512
    
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)
    
        self.center = nn.Sequential(
            ConvBnRelu2d(24, 24, kernel_size=3, padding=1),
        )
    
    
    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        out = self.center(out)
        out = self.up1(out, down1)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out
    
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class Discriminator_Cycle(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator_Cycle, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 64, depth 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (32, 32, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (16, 16, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (8, 8, 512)
        
        # Classification layer
        self.conv5 = conv(conv_dim*8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        # last, classification layer
        out = self.conv5(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 [3x256x256]
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 [256x128x128]
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 [512x64x64]
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 [1024x32x32]
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4 
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, 
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_WGAN(nn.Module):
    def __init__(self, DIM=64):
        self.DIM = DIM
        super(Discriminator_WGAN, self).__init__()
        self.main = nn.Sequential(
            # input = 3, output = 64
            nn.Conv2d(in_channels=3, out_channels=DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 64, output = 128
            nn.Conv2d(in_channels=DIM, out_channels=2 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 128, output = 256
            nn.Conv2d(in_channels=2 * DIM, out_channels=4 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output
  
# class PerceptualLoss():
	
# 	def contentFunc(self):
# 		conv_3_3_layer = 14
# 		cnn = models.vgg19(pretrained=True).features
# 		cnn = cnn.cuda()
# 		model = nn.Sequential()
# 		model = model.cuda()
# 		for i,layer in enumerate(list(cnn)):
# 			model.add_module(str(i),layer)
# 			if i == conv_3_3_layer:
# 				break
# 		return model
		
# 	def __init__(self, loss):
# 		self.criterion = loss
# 		self.contentFunc = self.contentFunc()
			
# 	def get_loss(self, fakeIm, realIm):
# 		f_fake = self.contentFunc.forward(fakeIm)
# 		f_real = self.contentFunc.forward(realIm)
# 		f_real_no_grad = f_real.detach()
# 		loss = self.criterion(f_fake, f_real_no_grad)
# 		return loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        #self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # TV Loss
        #tv_loss = self.tv_loss(out_images)
        return perception_loss