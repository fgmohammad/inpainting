# Credits https://github.com/MADF-inpainting/Pytorch-MADF

import torch
import torch.nn as nn


class FullModel(nn.Module):
  def __init__(self, model, extractor, args):
    super(FullModel, self).__init__()
    self.model = model
    self.extractor = extractor
    self.args = args

  def forward(self, image, mask, gt):
    input = image
    outputs = self.model(image, mask)
    
    feats = []
    comp_feats = []
    if self.args.use_incremental_supervision:
        if len(outputs) == 1:
            start = 0
        else:
            start = 1
    else:
        start = len(outputs) - 1

    for i in range(start, len(outputs)):
        output = outputs[i]
        output_comp = mask * input + (1 - mask) * output
        feats.append(self.extractor(output))
        comp_feats.append(self.extractor(output_comp))

    feat_gt = self.extractor(gt)
    return outputs, feats, comp_feats, feat_gt
    

def DataParallel_withLoss(model, extractor, args, **kwargs):
    model=FullModel(model, extractor, args)
    if 'device_ids' in kwargs.keys():
      device_ids=kwargs['device_ids']
    else:
      device_ids=None
    if 'output_device' in kwargs.keys():
      output_device=kwargs['output_device']
    else:
      output_device=None
    if 'cuda' in kwargs.keys():
      cudaID=kwargs['cuda'] 
      model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
      if torch.cuda.is_available():
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
      else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device)
    return model
