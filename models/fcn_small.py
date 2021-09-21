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

class FCN_(nn.Module):

    def __init__(self, backbone, num_class, dim):
        super(FCN_, self).__init__()
        self.backbone = backbone

        self.num_class = num_class
        self.dim = dim

        self.proj    = nn.Conv2d(in_channels=2048, out_channels=dim, kernel_size=3, padding=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=dim, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, num_class, kernel_size=1)


    def forward(self, x):

        input_shape = x.shape[-2:] 
        bs = x.size(0)
        #print('aqui')
        score = self.backbone(x)
        score = score['feat4']
        score = self.proj(score)
        features = score
        score = self.bn1(self.relu(self.deconv1(score)))
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)     

        return score