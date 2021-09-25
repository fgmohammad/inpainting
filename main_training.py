import numpy as np

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












###################################################################
##########                    SAMPLER                    ##########
###################################################################
class CustomSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return self.num_samples

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while i<self.num_samples:
            yield order[i]
            i += 1
######################################################################




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
    return model, optimizer, checkpoint['epoch']
###########################################################################













parser = argparse.ArgumentParser()
# training options
parser.add_argument('--image_root', type=str)     #root FOR IMAGES/MAPS
parser.add_argument('--masks_root', type=str)     #root FOR MASKS
parser.add_argument('--save_dir', type=str)       #save_dir TO SAVE MODEL CHECKPOINTS
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_threads', type=int, default=14)
parser.add_argument('--save_model_interval', type=int, default=5)     #SAVE MODEL CHECKPOINT EVERY save_model_interval
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=str)     #PATH TO THE MODEL CHECKPOINT TO RESUME TRAINING
parser.add_argument('--use_incremental_supervision', action='store_true')
parser.add_argument('--n_refinement_D', type=int, default=2)
parser.add_argument('--valid_weight', type=float, default=1.0)
parser.add_argument('--hole_weight', type=float, default=6.0)
parser.add_argument('--tv_weight', type=float, default=0.1)
parser.add_argument('--prc_weight', type=float, default=0.05)
parser.add_argument('--style_weight', type=float, default=120.0)
args = parser.parse_args()




print (f'Start: {datetime.datetime.now()}', flush=True)

################################################################
##########               LOAD TRAIN SET               ##########
################################################################
path_images = args.image_root + 'train_maps/'
path_masks = args.masks_root + f'train_masks/'
print (f'train map: {path_images}')
print (f'train masks: {path_masks}')
transform_img = transforms.Compose([log_transform(), transforms.Normalize(opt.MEAN, opt.STD)])
train_set = DataSet (path_images, path_masks, transform_img)
train_loader = data.DataLoader(dataset=train_set, 
                               batch_size=args.batch_size, 
                               num_workers=args.n_threads,
                               sampler=CustomSampler(len(train_set)))

print (f'Train Set Loaded: {datetime.datetime.now()}', flush=True)




#####################################################################
##########               LOAD VALIDATION SET               ##########
#####################################################################
path_images = args.image_root + 'validation_maps/'
path_masks = args.masks_root + f'validation_masks/'
transform_img = transforms.Compose([log_transform(), transforms.Normalize(opt.MEAN, opt.STD)])
validation_set = DataSet (path_images, path_masks, transform_img)
validation_loader = data.DataLoader(dataset=validation_set, 
                               batch_size=args.batch_size, 
                               num_workers=args.n_threads,
                               sampler=CustomSampler(len(validation_set)))

print (f'Validation Set Loaded: {datetime.datetime.now()}', flush=True)




device = set_device()

#######################################################
##########               MODEL               ##########
#######################################################
model = MADFNet(layer_size=7, args=args).to(device)
torch.backends.cudnn.benchmark = True
model = DataParallel_withLoss(model, VGG16FeatureExtractor(), args)
print (f'Model Initialized: {datetime.datetime.now()}', flush=True)



##########          OPTIMIZER          ##########
lr = args.lr
gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
print (f'Optimizer Initializeds: {datetime.datetime.now()}', flush=True)


##########          SCHEDULER          ##########
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, factor=0.1, patience=5)
print (f'Scheduler Initialized: {datetime.datetime.now()}', flush=True)





##########          LOSS          ##########
criterion = InpaintingLoss(VGG16FeatureExtractor(), args).to(device)
print (f'Loss Initialized: {datetime.datetime.now()}', flush=True)







##########          LOAD MODEL IF REQUESTED          ##########
if args.resume:
    model, gen_optimizer, epoch_in = load_ckpt(args.resume, model, gen_optimizer, device)
    print (f'Model Loaded: {datetime.datetime.now()}', flush=True)
else:
    epoch_in = 0


print (f'Epoch_in: {epoch_in}')





##########          TRAINING          ##########
for epoch in range (epoch_in, args.max_epochs):
    #####     TRAIN ONE EPOCH AT A TIME
    train_loss, validation_loss = training (train_loader, validation_loader, 
                                            criterion, 
                                            gen_optimizer, scheduler, model, 
                                            args, device=device)
    
    #####     SAVE TRAIN LOSS
    path_loss = f'losses_madf/train_loss_madf_camels_InpaintingLoss.dat'
    if not os.path.isfile(path_loss):
        with open (path_loss, 'a') as ofile:
            ofile.write ('#\t\tVALID\t\t\tHOLE\t\t\tPRC\t\t\tSTYLE\t\t\tTV\t\t\tTOTAL\n')
    with open (path_loss, 'a') as ofile:
        ofile.write(f'{epoch}\t\t\t{train_loss["valid"]}\t{train_loss["hole"]}\t{train_loss["prc"]}\t{train_loss["style"]}\t{train_loss["tv"]}\t{train_loss["total"]}\n')
    
    #####     SAVE VALIDATION LOSS
    path_loss = f'losses_madf/validation_loss_madf_camels_InpaintingLoss.dat'
    if not os.path.isfile(path_loss):
        with open (path_loss, 'a') as ofile:
            ofile.write ('#\t\t\tVALID\t\t\t HOLE\t\t\tPRC\t\t\tSTYLE\t\t\tTV\t\t\tTOTAL\n')
    with open (path_loss, 'a') as ofile:
        ofile.write(f'{epoch}\t\t\t{validation_loss["valid"]}\t{validation_loss["hole"]}\t{validation_loss["prc"]}\t{validation_loss["style"]}\t{validation_loss["tv"]}\t{validation_loss["total"]}\n')

    if (((epoch)%args.save_model_interval)==0):
        model_path = args.save_dir + f'madf_camels_InpaintingLoss_epoch-{epoch}.pth'
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': gen_optimizer.state_dict(),
        'loss': train_loss['total'],
        }, model_path)

    print (f'Epoch: {epoch:05d}\t\tTrain_Loss = {format(train_loss["total"], ".5f")}\t\tValidation_Loss = {format(validation_loss["total"], ".5f")}\t\tlr = {gen_optimizer.param_groups[0]["lr"]}\t\t{datetime.datetime.now()}', flush=True)
