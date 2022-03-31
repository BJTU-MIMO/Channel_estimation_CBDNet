import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as data
import math


def svd_orthogonalization(lyr):
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        weights = lyr.weight.data.clone()
        c_out, c_in, f1, f2 = weights.size()
        dtype = lyr.weight.data.type()
        weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)
        weights = weights.cpu().numpy()
        mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
        weights = np.dot(mat_u, mat_vh)
        lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).type(dtype)

    else:
        pass


def weights_init_kaiming(lyr):
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.25, 0.25)
        nn.init.constant_(lyr.bias.data, 0.0)
