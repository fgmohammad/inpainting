import numpy as np

import random

import os
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data

import datetime

from dataset import DataSet
from custom_transformations import log_transform

import opt

from training import *
from loss import InpaintingLoss

from madf import MADFNet
from madf import VGG16FeatureExtractor

from data_parallel import DataParallel_withLoss

from image_utils import unnormalize

from astropy.io import fits
from astropy.table import Table,Column







######################################################################
##########                    SET DEVICE                    ##########
######################################################################
def set_device():
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        print ('Device: GPU')
    else:
        device = torch.device('cpu')
        print ('Device: CPU')
    return device
######################################################################




###########################################################################
##########                    LOAD CHECKPOINT                    ##########
###########################################################################
def load_ckpt(checkpoint_fpath, model, optimizer, device):
    if device:
        checkpoint = torch.load(checkpoint_fpath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #loss = checkpoint['loss']
    return model, optimizer, checkpoint['epoch']
###########################################################################


















parser = argparse.ArgumentParser()
# training options
parser.add_argument('--image_path', type=str)
parser.add_argument('--mask_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--n_threads', type=int, default=1)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=130)
parser.add_argument('--use_incremental_supervision', action='store_true')
parser.add_argument('--n_refinement_D', type=int, default=2)
args = parser.parse_args()




print (f'Start: {datetime.datetime.now()}', flush=True)

###########################################################
##########               LOAD DATA               ##########
###########################################################
transform_img = transforms.Compose([log_transform(), transforms.Normalize(opt.MEAN, opt.STD)])



device = set_device()



#######################################################
##########               MODEL               ##########
#######################################################
model = MADFNet(layer_size=7, args=args).to(device)
model = DataParallel_withLoss(model, VGG16FeatureExtractor(), args)
print (f'Model Initialized: {datetime.datetime.now()}', flush=True)


##########          OPTIMIZER          ##########
lr = 0.0002
gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
print (f'Optimizer Initializeds: {datetime.datetime.now()}', flush=True)

##########          SCHEDULER          ##########
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, factor=0.1)
print (f'Scheduler Initialized: {datetime.datetime.now()}', flush=True)

##########          LOSS          ##########
criterion = InpaintingLoss(VGG16FeatureExtractor(), args).to(device)
print (f'Loss Initialized: {datetime.datetime.now()}', flush=True)

##########          LOAD MODEL          ##########
model, gen_optimizer, epoch = load_ckpt (args.model_path, model, gen_optimizer, device)
print (f'Model Loaded: {datetime.datetime.now()}', flush=True)





model.eval()

gt = torch.from_numpy(np.load(args.image_path)).type(torch.FloatTensor).repeat(3,1,1)
gt = transform_img(gt)
gt = torch.reshape(gt, (1, gt.shape[0], gt.shape[1], gt.shape[2]))

    
#####     LOAD MASK    
mask = torch.from_numpy(np.load(args.mask_path)).type(torch.FloatTensor).repeat(3,1,1)
mask = torch.reshape(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2]))
inputs = gt*mask


inputs, mask, gt = inputs.to(device), mask.to(device), gt.to(device)

yhat = model(inputs, mask, gt)
output = yhat[0][-1]


inputs = 10.**(unnormalize(inputs, opt.MEAN, opt.STD, device))
output = 10.**(unnormalize(output, opt.MEAN, opt.STD, device))
gt = 10.**(unnormalize(gt, opt.MEAN, opt.STD, device))


output = torch.squeeze(output[:,0,:,:]).detach().cpu().numpy()
mask = torch.squeeze(mask[:,0,:,:]).detach().cpu().numpy()
gt = torch.squeeze(gt[:,0,:,:]).detach().cpu().numpy()

data = Table()
img_out = Column (output, name='OUTPUT', dtype='float')
img_gt = Column (gt, name='GT', dtype='float')
img_msk = Column (mask, name='MASK', dtype='int')
data.add_columns([img_gt, img_msk, img_out])

path_out = args.save_dir + f'predictions_madf.fits'
data.write (path_out, format='fits')

