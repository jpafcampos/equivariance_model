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

class ResViT_timm(nn.Module):

    def __init__(self, pretrained_net, num_class):
        super(ResViT_timm, self).__init__()

        self.n_class = num_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        #print(x.size())
        bs = x.size(0)
        img_size = x.shape[-2:]   
        x = self.pretrained_net(x)
        #print(x.size())
        
        score = torch.reshape(x, (bs, img_size[1]//16, img_size[0]//16, 768))
        score = score.view(
            score.size(0),
            24,
            24,
            768
        )
        score = score.permute(0, 3, 1, 2).contiguous()

        score = self.bn1(self.relu(self.deconv1(score)))    
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.classifier(score)                    

        return score  # size=(N, n_class, x.H/1, x.W/1)  