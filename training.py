import torch
import torch.nn as nn
import torch.nn.functional as F




def training (train_loader, validation_loader, criterion, optimizer, scheduler, model, args, device=None):
        
    
    ##########               TRAINING               ##########
    train_loss = {}
    for x, mask, gt in train_loader:
        if device:
            x, mask, gt = x.to(device), mask.to(device), gt.to(device)        
        model.train()
        output, feats, comp_feats, feat_gt = model(x, mask, gt)
        loss_dict = criterion(x, mask, output, gt, comp_feats, feats, feat_gt)
        loss = 0.
        for key in loss_dict:
            coeff = getattr(args, key+'_weight')
            value = coeff*loss_dict[key]
            loss+=value
            train_loss[key] = loss_dict[key].detach().data.item()
        train_loss['total'] = loss.detach().data.item()
        del loss_dict
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    ##########               VALIDATION SET LOSS               ##########
    model.eval()
    iterations = 0
    _validation_loss = 0.
    validation_loss = {}
    keys = ['tv', 'prc', 'hole', 'style', 'valid']
    for key in keys:
        validation_loss[key] = 0.
    for x, mask, gt in validation_loader:
        if device:
            x, mask, gt = x.to(device), mask.to(device), gt.to(device)
        output, feats, comp_feats, feat_gt = model(x, mask, gt)
        loss_dict = criterion(x, mask, output, gt, comp_feats, feats, feat_gt)
        for key in loss_dict:
            coeff = getattr(args, key+'_weight')
            value = coeff*loss_dict[key]
            _validation_loss+=value.detach().data.item()
            validation_loss[key]+=loss_dict[key].detach().data.item()
        del loss_dict
        iterations+=1

    validation_loss['total'] = _validation_loss
    for key in validation_loss:
        validation_loss[key]/=float(iterations)
    
    if scheduler:
        scheduler.step(train_loss['total'])
    return train_loss, validation_loss
