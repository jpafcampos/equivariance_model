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

class Resvit(nn.Module):

    def __init__(self, backbone, num_class, dim=2048, depth=1, heads=8, mlp_dim=3072):
        super(Resvit, self).__init__()
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

        self.num_class = num_class

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)

        self.classifier = nn.Conv2d(128, num_class, kernel_size=1)


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
        #print("score permute", score.size())
        score = self.bn1(self.relu(self.deconv1(score)))
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)     

        return score   