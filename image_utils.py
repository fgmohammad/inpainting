import torch 



def unnormalize(x, MEAN, STD, device):
    """
    UNDO NORMALIZATION
    MEAN, STD: Parameters used to normalize to 0 mean and 1 variance
    """
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD).to(device) + torch.Tensor(MEAN).to(device)
    x = x.transpose(1, 3)
    return x
