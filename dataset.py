import numpy as np

import os

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms








class DataSet(Dataset):
    def __init__ (self, img_root, mask_root, transform_img):
        self.img_root = img_root
        self.mask_root = mask_root
        
        self.transform_img = transform_img
        
        self.img_paths = []
        for root, _, fnames in os.walk(img_root):
            for fname in fnames:
                path = os.path.join(root, fname)
                self.img_paths.append(path)
       
        self.mask_paths = []
        for root, _, fnames in os.walk(mask_root):
            for fname in fnames:
                path = os.path.join(root, fname)
                self.mask_paths.append(path)
        
        self.N_mask = len(self.mask_paths)

    def __getitem__ (self, index):
        gt_map = torch.from_numpy(np.load(self.img_paths[index])).type(torch.FloatTensor).repeat(3,1,1)
        gt_map = self.transform_img(gt_map)
        
        #####     CHOOSE A RANDOM MASK FROM THE POOL OF N_mask BINARY MASKS
        idx = random.randint(0, self.N_mask - 1)
        mask = torch.from_numpy(np.load(self.mask_paths[idx])).type(torch.FloatTensor).repeat(3,1,1)
        return gt_map*mask, mask, gt_map
    
    def __len__(self):
        return (len(self.img_paths))
