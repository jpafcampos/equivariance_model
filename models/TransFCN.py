from __future__ import print_function

import os.path as osp
import copy
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.utils import load_state_dict_from_url
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from .fcn32s import get_upsampling_weight
import sys
sys.path.insert(1, '../utils')
import utils as U
import vit
import line_profiler

class TransFCN(nn.Module):

    def __init__(self, backbone, transformer, num_class):
        super(TransFCN, self).__init__()
        self.backbone = backbone
        self.channel_reduction = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, padding=1)
        self.transformer = transformer
        self.num_class = num_class
        self.headconv1 = nn.Conv2d(768, 512, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.headconv2 = nn.Conv2d(512, num_class, 1)

    #@profile
    def forward(self, x):

        input_shape = x.shape[-2:] 
        bs = x.size(0)
        #print('aqui')
        score = self.backbone(x)
        score = score['out']
        score = self.channel_reduction(score) 
        img_size = score.shape[-2:]
        score = self.transformer(score)
        #print(img_size)
        score = torch.reshape(score, (bs, img_size[1], img_size[0], 768))
        score = torch.transpose(score, 1, 3)

        score = F.interpolate(score, size=input_shape, mode='bilinear', align_corners=False)
        #print(score.size())
        score = self.headconv1(score)
        score = self.bn(score)
        score = self.relu(score)
        score = self.dropout(score)
        score = self.headconv2(score)


        return score  # size=(N, n_class, x.H/1, x.W/1)   
