import torch
import torch.nn as nn



class log_transform(nn.Module):
    def __init__ (self):
        super().__init__()
    def __call__ (self, img):
        return torch.log10(img)
