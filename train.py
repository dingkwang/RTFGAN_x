# By Yuxiang Sun, Dec. 4, 2019
# Email: sun.yuxiang@outlook.com

# Edited by Sanjeev J. Koppal Spring 2021

import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as N
from torch.autograd import Variable
import torchvision.utils as vutils
from PIL import Image
import torchvision.models as models


import logging
import logging.config

# THESE ARE TWO CLASSESTHAT ARE  USEFUL FORREADING DATA FROM THE DATASET
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset

from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import RTFNet

import torch.optim as optim

# This is from the GAN github repo
from ganmodels import Generator, Discriminator, LossNetwork

import utils

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 

# HERE IS THE BATCH SIZE
parser.add_argument('--batch_size', '-b', type=int, default=5) 
# LEARNING RATE
parser.add_argument('--lr_start', '-ls', type=float, default=0.0001)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='/blue/eel6935/share/GAN/RTFNet/dataset/')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

def weights_init(neural_net):
    """ Initializes the weights of the neural network
    :param neural_net: (De-)Convolutional Neural Network where weights should be initialized
    """
    classname = neural_net.__class__.__name__
    if classname.find('Conv') != -1:
        neural_net.weight.data.normal_(0, 2e-2)
    elif classname.find('BatchNorm') != -1:
        neural_net.weight.data.normal_(1, 2e-2)
        neural_net.bias.data.fill_(0)

# PROJECT1: TRY DIFFERENT LOSSES HERE
#criterion = N.BCELoss()
#criterion3 = N.MSELoss()
#criterion2 = N.L1Loss()
# TRAIN DISCRIMINATOR BY ITSELF FOR A BIT
#INITEPO = 30
#INITEPO2 = 8

# FROM DINGKANG
# PROJECT1: TRY DIFFERENT LOSSES HERE
# criterion = N.BCELoss() Original 
criterion = N.BCEWithLogitsLoss()
criterion3 = N.MSELoss()
criterion2 = N.L1Loss()
# criterion2 = N.CrossEntropyLoss() NOT working 
# TRAIN DISCRIMINATOR BY ITSELF FOR A BIT
#INITEPO = 150
INITEPO = 80
#INITEPO2 = 8

#FLAG FOR USING WASSSERSTEIN ERROR
WASSER = 1


def train(epo, net_generator, net_discriminator, train_loader, optimizer_generator, optimizer_discriminator,loss_network):#, optimizer):
    #model.train()
    net_generator.train()
    net_discriminator.train()
    criterion.cuda(args.gpu)

    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time() # time.time() returns the current time
        #optimizer.zero_grad()

        #print(images.size())
 
        # THIS IS WHERE THE IMAGES ARE PUMPED THROUGH THE MODEL AND YOU GET AN OUTPUT
        # logits = model(images)
        # LOSS IS CALCULATED HERE
        # loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function

	# We replace the above with GAN losses from the GAN github repo
        # SPLITTING RGB AND THERMAL

        rgb = images[:,:3,:,:]
        thermal = images[:,3:,:,:]

        # FIXED THE CRAZy BuG!!
        thermalbig = rgb.clone().detach()
        fakebig = rgb.clone().detach()

        thermalbig[:,0,:,:] = images[:,3,:,:] 
        thermalbig[:,1,:,:] = images[:,3,:,:] 
        thermalbig[:,2,:,:] = images[:,3,:,:] 

        #print(images.size())
        #print(rgb.size())
        #print(thermal.size())

        ## 1. update weights of discriminator in respect of the gradient
        net_discriminator.zero_grad()

        ## train discriminiator on real images
        #real_img = Variable(data[0])
        batch_size = thermal.size()[0]
        ones = Variable(torch.ones(batch_size))
        ones.cuda(args.gpu)
        output = net_discriminator.forward(thermal)
        output.cuda(args.gpu)

        #FOR WASSERSTEIN -- DO NOT USE THE "ONES" and "ZEROS", just use the error directly
        err_D_real = output.cuda()

        # loss network takes normalized data only
        thermalbig = thermalbig.detach()
        maxfakeimage = np.amax(np.asarray(thermalbig.cpu()))
        thermalbig = np.divide(thermalbig.cpu(),maxfakeimage)

        thermal_pl = loss_network.forward(thermalbig.cuda())
        #thermal_pl = tmp.relu3_3
        #thermal_pl = tmp.relu4_3
        #print(thermal_pl.size())

        ogray = rgb.cuda()
        #print(gray.size())
        gray = torch.mean(ogray,1,False)
        graycomp = torch.mean(ogray,1,True)
        #print(gray.size())

        #firstfakeimage = np.squeeze(fake_img[1,:,:])
        firstfakeimage2 = np.squeeze(gray[1,:,:])
        firstfakeimage2 = firstfakeimage2.detach()
        maxfakeimage2 = np.amax(np.asarray(firstfakeimage2.cpu()))
        firstfakeimage2 = np.divide(firstfakeimage2.cpu(),maxfakeimage2)
        firstfakeimage2 = firstfakeimage2*255.0
        firstfakeimage2 = firstfakeimage2.cpu()
        #print(tmp)
       # Save some images
        if it == 1:
          image_path = 'gray_{}_{}.jpg'.format(epo, it)
          image_path = os.path.join('./ganruns', image_path)
          fim = Image.fromarray(np.asarray(firstfakeimage2).astype(np.uint8))
          fim.save(image_path)

        #print(output.size())
        #print(ones.size())

        real_img_error = criterion(output.cuda(), ones.cuda()).cuda()
        #print(rgb.size())

        ## train discriminiator on fake images
        fake_img = net_generator.forward(rgb)
        fake_img_comp = np.squeeze(fake_img)
        #print(fake_img.size())

        fakebig[:,0,:,:] = fake_img[:,0,:,:] 
        fakebig[:,1,:,:] = fake_img[:,0,:,:] 
        fakebig[:,2,:,:] = fake_img[:,0,:,:] 

        # loss network takes normalized data only
        fakebig = fakebig.detach()
        maxfakeimage = np.amax(np.asarray(fakebig.cpu()))
        fakebig = np.divide(fakebig.cpu(),maxfakeimage)

        fake_pl = loss_network.forward(fakebig.cuda())
        #fake_pl = tmp.relu3_3
        #fake_pl = tmp.relu4_3
        #print(fake_pl.size())


        zeros = Variable(torch.zeros(batch_size))

        # detach the gradient from generator (saves computation)
        if epo > INITEPO:
           #print('comparing to fake')
           output = net_discriminator.forward(fake_img.detach())
        else:
           output = net_discriminator.forward(graycomp.detach())

        #FOR WASSERSTEIN -- DO NOT USE THE "ONES" and "ZEROS", just use the error directly
        err_D_fake = output.cuda()

        fake_img_error = criterion(output.cuda(), zeros.cuda()).cuda()
        

        ## backpropagate total error
        descriminator_error = real_img_error + fake_img_error
        
        ## compute Gradient Penalty
        device = torch.device('cuda')
        gradient_penalty = utils.calc_gradient_penalty(netD=net_discriminator, real_data=thermal, fake_data=fake_img, BATCH_SIZE=batch_size,device=device)

        if WASSER:
           descriminator_error = err_D_real.norm() - err_D_fake.norm() + gradient_penalty

        descriminator_error.backward()
        optimizer_discriminator.step()

   
        if epo > INITEPO:
           ## 2. update weights of generator
           # now we keep the gradient so we can update the weights of the generator
           output = net_discriminator.forward(fake_img)

           # WASSERSTEIN ERROR
           err_G = output.cuda()

           # For 10 epochs, the generator basically becomes a gray image creator 
           if epo > INITEPO:
             #print('GAN opt')
             generator_error = criterion(output.cuda(), ones.cuda()).cuda()
             if WASSER:
                generator_error = err_G.norm()
             #if net_generator.optimflag == 0:
             #   print('reset of opt')
             #   net_generator.optimflag = 1
             #   for g in optimizer_generator.param_groups:
             #    g['lr'] = 0.00001
             #   for g in optimizer_discriminator.param_groups:
             #    g['lr'] = 0.00001

                #optimizer_generator = optim.Adam(net_generator.parameters(),
                #                     0.000001)
                                     
                #optimizer_discriminator = optim.Adam(net_discriminator.parameters(),
                #                         0.000001)
        else:
           #print(criterion2(fake_img.cuda(),thermal.cuda()).cuda())
           #print(criterion3(fake_pl.cuda(),thermal_pl.cuda()))
           tmp1 = criterion3(fake_pl.relu1_2.cuda(),thermal_pl.relu1_2.cuda())
           tmp2 = criterion3(fake_pl.relu2_2.cuda(),thermal_pl.relu2_2.cuda())
           tmp3 = criterion3(fake_pl.relu3_3.cuda(),thermal_pl.relu3_3.cuda())
           tmp4 = criterion3(fake_pl.relu4_3.cuda(),thermal_pl.relu4_3.cuda())
           ploss = tmp1 + tmp2 + tmp3 + tmp4
           #print(ploss)
           generator_error = criterion2(fake_img.cuda(),thermal.cuda()).cuda() + ploss.cuda()
						
        net_generator.zero_grad()
        generator_error.backward()
        optimizer_generator.step()


        #firstfakeimage = np.squeeze(fake_img[1,:,:])
        #firstfakeimage2 = np.squeeze(gray[1,:,:])
        #firstfakeimage2 = firstfakeimage2.detach()
        #maxfakeimage2 = np.amax(np.asarray(firstfakeimage2.cpu()))
        #firstfakeimage2 = np.divide(firstfakeimage2.cpu(),maxfakeimage2)
        #firstfakeimage2 = firstfakeimage2*255.0
        #firstfakeimage2 = firstfakeimage2.cpu()
        #print(tmp)


       # Save some images
        #if it == 1:
        #  image_path = 'gray_{}_{}.jpg'.format(epo, it)
        #  image_path = os.path.join('./ganruns', image_path)
        #  fim = Image.fromarray(np.asarray(firstfakeimage2).astype(np.uint8))
        #  fim.save(image_path)

        #firstfakeimage = np.squeeze(fake_img[1,:,:])
        firstfakeimage = np.squeeze(thermal[1,:,:])
        firstfakeimage = firstfakeimage.detach()
        maxfakeimage = np.amax(np.asarray(firstfakeimage.cpu()))
        firstfakeimage = np.divide(firstfakeimage.cpu(),maxfakeimage)
        firstfakeimage = firstfakeimage*255.0
        firstfakeimage = firstfakeimage.cpu()
        #print(tmp)
       # Save some images
        if it == 1:
          image_path = 'thermal_{}_{}.jpg'.format(epo, it)
          image_path = os.path.join('./ganruns', image_path)
          fim = Image.fromarray(np.asarray(firstfakeimage).astype(np.uint8))
          fim.save(image_path)

        firstfakeimage = np.squeeze(fake_img[1,:,:])
        #firstfakeimage = np.squeeze(gray[1,:,:])
        firstfakeimage = firstfakeimage.detach()
        maxfakeimage = np.amax(np.asarray(firstfakeimage.cpu()))
        firstfakeimage = np.divide(firstfakeimage.cpu(),maxfakeimage)
        firstfakeimage = firstfakeimage*255.0
        firstfakeimage = firstfakeimage.cpu()
        #print(tmp)
       # Save some images
        if it == 1:
          image_path = 'fake_{}_{}.jpg'.format(epo, it)
          image_path = os.path.join('./ganruns', image_path)
          fim = Image.fromarray(np.asarray(firstfakeimage).astype(np.uint8))
          fim.save(image_path)


        # loss.backward()
        # optimizer.step()

        lr_this_epo=0
        lr_this_epo1=0
        for param_group in optimizer_generator.param_groups:
            lr_this_epo = param_group['lr']
        for param_group in optimizer_discriminator.param_groups:
            lr_this_epo1 = param_group['lr']

        print('Generator: epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
            % (epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(generator_error),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        print('Discriminator: epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
            % (epo, args.epoch_max, it+1, len(train_loader), lr_this_epo1, len(names)/(time.time()-start_t), float(descriminator_error),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))


def save_model(state_dict, name, epoch):
    """Saves the trained neural net, optimizer and epoch
    :param state_dict: Dictionary of states from the (De-)Convolutional Neural Network, optimizer & epoch
    :param name: Name of the Neural Network
    :param epoch: Current epoch
    """
    logging.getLogger(__name__).info(
        'Saving trained model {} at epoch {}.'.format(name, epoch))
    if not os.path.exists('model'):
        os.makedirs('model')
    model_name = 'dcgan_{}_{}.pth'.format(name, epoch)
    model_path = os.path.join('./ganruns', model_name)
    torch.save(state_dict, model_path)

LOADMODEL = 0
EPOCHTOLOAD = 151


# THE MAIN FUNCTION THAT IS RUN WHEN YOU RUN THE CODE
if __name__ == '__main__':
   
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # THE BLACK BOX, THE NEURAL NETWORK
    #model = eval(args.model_name)(n_class=args.n_class)
    # Initialize the generator and discriminator
    print('Create Generator and Discriminator and initialize weights.')


    # LOAD PREVIOUSLY TRAINED WEIGHTS
    if LOADMODEL == 1:
      net_generator = Generator()
      net_discriminator = Discriminator()

      model_name = '{}/dcgan_{}_{}.pth'.format("./ganruns_exp3","generator", EPOCHTOLOAD)
      # pretrained_weight = torch.load(model_name, map_location = lambda storage, loc: storage.cuda(args.gpu))
      checkpoint = torch.load(model_name, map_location = lambda storage, loc: storage.cuda(args.gpu))
      net_generator.load_state_dict(checkpoint['state_dict'])
      optimizer_generator = optim.Adam(net_generator.parameters(),args.lr_start,betas=(.5, .999))
      optimizer_generator.load_state_dict(checkpoint['optimizer'])
      for state in optimizer_generator.state.values():
          for k, v in state.items():
              if isinstance(v, torch.Tensor):
                  state[k] = v.cuda()
      newepo = checkpoint['epoch']



      model_name = '{}/dcgan_{}_{}.pth'.format("./ganruns_exp3","discriminator", EPOCHTOLOAD)
      checkpoint = torch.load(model_name, map_location = lambda storage, loc: storage.cuda(args.gpu))
      net_discriminator.load_state_dict(checkpoint['state_dict'])
      optimizer_discriminator = optim.Adam(net_discriminator.parameters(),args.lr_start,betas=(.5, .999))
      optimizer_discriminator.load_state_dict(checkpoint['optimizer'])
      for state in optimizer_discriminator.state.values():
          for k, v in state.items():
              if isinstance(v, torch.Tensor):
                  state[k] = v.cuda()
      newepo = checkpoint['epoch']

      net_generator.cuda()
      net_discriminator.cuda()
      args.epoch_from = newepo 

    else:
       net_generator = Generator().apply(weights_init)
       net_discriminator = Discriminator().apply(weights_init)
       optimizer_generator = optim.Adam(net_generator.parameters(),args.lr_start,betas=(.5, .999))
       optimizer_discriminator = optim.Adam(net_discriminator.parameters(),args.lr_start,betas=(.5, .999))


    # START THE SGD
    #if args.gpu >= 0: model.cuda(args.gpu)
    if args.gpu >= 0: 
        net_generator.cuda(args.gpu)
        net_discriminator.cuda(args.gpu)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("./ganruns"):
        shutil.rmtree("./ganruns")
    #weight_dir = os.path.join("./ganruns", args.model_name)
    weight_dir = os.path.join("./ganruns")
    os.makedirs(weight_dir)
    #os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./ganruns")
    #os.chmod("./runs/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    #os.chmod("./runs", stat.S_IRWXO) 

    print('training on GPU #%d with pytorch' % (args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    # DATASETS FOR TRAINING, TESTING and VAL
    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    #val_dataset  = MF_dataset(data_dir=args.data_dir, split='val')
    #test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    #val_loader  = DataLoader(
    #    dataset     = val_dataset,
    #    batch_size  = args.batch_size,
    #    shuffle     = False,
    #    num_workers = args.num_workers,
    #    pin_memory  = True,
    #    drop_last   = False
    #)
    #test_loader = DataLoader(
    #    dataset      = test_dataset,
    #    batch_size   = args.batch_size,
    #    shuffle      = False,
    #    num_workers  = args.num_workers,
    #    pin_memory   = True,
    #    drop_last    = False
    #)
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}

    vgg_model = models.vgg16(pretrained=True)
    vgg_model.cuda()
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain epo #%s begin...' % (epo))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, net_generator,net_discriminator, train_loader, optimizer_generator, optimizer_discriminator,loss_network)#, optimizer)
        #validation(epo, model, val_loader)

        #checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        #print('saving check point %s: ' % checkpoint_model_file)
        #torch.save(model.state_dict(), checkpoint_model_file)
        if (epo % 10 == 0):
                save_model({
                    'epoch': epo + 1,
                    'state_dict': net_generator.state_dict(),
                    'optimizer': optimizer_generator.state_dict()
                }, 'generator', int(epo + 1))
                save_model({
                    'epoch': epo + 1,
                    'state_dict': net_discriminator.state_dict(),
                    'optimizer': optimizer_discriminator.state_dict()
                }, 'discriminator', int(epo + 1))

        #testing(epo, model, test_loader)
        #scheduler.step() # if using pytorch 1.1 or above, please put this statement here
