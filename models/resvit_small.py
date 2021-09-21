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

    def __init__(self, backbone, num_class, dim, depth, heads, mlp_dim, ff=True):
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
            channels = dim,
            dim_head = self.dim_head,
            dropout = 0.1,
            emb_dropout = 0.1,
            ff = ff
        )

        self.num_class = num_class

        self.proj    = nn.Conv2d(in_channels=2048, out_channels=dim, kernel_size=3, padding=1)
        #self.proj    = nn.Conv2d(in_channels=2048, out_channels=dim, kernel_size=1, stride=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(dim, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, num_class, kernel_size=1)


    def forward(self, x):

        input_shape = x.shape[-2:] 
        bs = x.size(0)
        score = self.backbone(x)
        score = score['feat4']
        img_size = score.shape[-2:]
        score = self.proj(score)
        score = self.transformer(score)
        score = torch.reshape(score, (bs, img_size[1], img_size[0], self.dim))
        score = score.view(
            score.size(0),
            64,
            64,
            self.dim
        )
        score = score.permute(0, 3, 1, 2).contiguous()
        features = score
        score = self.bn1(self.relu(self.deconv1(score)))
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)     

        return score


#resnet50_dilation = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
#backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
#model = Resvit(backbone=backbone_dilation, num_class=5, dim=768, depth=1, heads=2, mlp_dim=3072, ff=True)
#
#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#print(count_parameters(model))
#print(model)