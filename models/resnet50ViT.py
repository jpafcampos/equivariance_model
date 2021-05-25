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




class ResViT(nn.Module):

    def __init__(self, pretrained_net, num_class, dim, depth, heads, batch_size, trans_img_size):
        super(ResViT, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.batch_size = batch_size
        self.trans_img_size = trans_img_size
        #Transformer unit (encoder)
        self.transformer = vit.ViT(
            image_size = trans_img_size,
            patch_size = 1,
            num_classes = 64, #not used
            dim = dim,
            depth = depth,    #number of encoders
            heads = heads,    #number of heads in self attention
            mlp_dim = 3072,   #hidden dimension in feedforward layer
            channels = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.channel_reduction = nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=3, padding=1)
        self.n_class = num_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        out_resnet = self.pretrained_net(x)
        #x1 = out_resnet['feat1']    #H/4   256 ch
        #x2 = out_resnet['feat2']    #H/8   512 ch
        x3 = out_resnet['feat3']    #H/16  1024 ch
        #x4 = out_resnet['feat4']    #H/32  2048 channels
        
        x3 = self.transformer(x3)
        x3 = torch.reshape(x3, (self.batch_size,self.dim,self.trans_img_size,self.trans_img_size))
        x3 = self.channel_reduction(x3)

        score = self.bn1(self.relu(self.deconv1(x3)))    
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = self.classifier(score)                    

        return score  # size=(N, n_class, x.H/1, x.W/1)                   

#resnet50 = models.resnet50(pretrained=True)
#new_m = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
#model = ResViT(pretrained_net=new_m, num_class=4, dim=768, depth=3, heads=6, batch_size = 8, trans_img_size=32)
#model.eval()
#pred = model(torch.rand(8, 3, 512, 512))
#print(pred.size())