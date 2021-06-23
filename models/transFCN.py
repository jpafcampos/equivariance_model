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

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

class TransFCN(nn.Module):

    def __init__(self, backbone, num_class, dim, depth, heads, mlp_dim):
        super(TransFCN, self).__init__()
        self.backbone = backbone
        self.dim = dim
        self.heads = heads
        self.dim_head = dim//heads
        self.transformer = vit.ViT(
            image_size = 64,
            patch_size = 1,
            num_classes = 64, #not used
            dim = dim,
            depth = depth,    #number of encoders
            heads = heads,    #number of heads in self attention
            mlp_dim = mlp_dim,   #hidden dimension in feedforward layer
            channels = 2048,
            dim_head = self.dim_head,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.proj = nn.Conv2d(in_channels=dim, out_channels=2048, kernel_size=3, padding=1)
        self.classifier = FCNHead(2048, num_class)
        self.num_class = num_class

    def forward(self, x):

        input_shape = x.shape[-2:] 
        bs = x.size(0)
        #print('aqui')
        score = self.backbone(x)
        score = score['feat4']
        #print(score.size())
        img_size = score.shape[-2:]
        score = self.transformer(score)
        #print(img_size)
        score = torch.reshape(score, (bs, img_size[1], img_size[0], self.dim))
        score = score.view(
            score.size(0),
            64,
            64,
            self.dim
        )
        score = score.permute(0, 3, 1, 2).contiguous()
        score = self.proj(score)
        
        score = self.classifier(score)
        score = F.interpolate(score, size=input_shape, mode='bilinear', align_corners=False)


        return score  # size=(N, n_class, x.H/1, x.W/1) 