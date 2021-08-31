from __future__ import print_function
import timm
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

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs

class VitBackbone(nn.Module):
    def __init__(self, vit_backbone):
        super(VitBackbone, self).__init__()
        self.vit_backbone = vit_backbone
    def forward(self, x):
        x = vit_backbone[0](x)
        x = vit_backbone[1](x)
        x0 = vit_backbone[2][0](x)
        x1 = vit_backbone[2][1](x0)
        x2 = vit_backbone[2][2](x1)
        x3 = vit_backbone[2][3](x2)
        x4 = vit_backbone[2][4](x3)
        x5 = vit_backbone[2][5](x4)
        x6 = vit_backbone[2][6](x5)
        x7 = vit_backbone[2][7](x6)
        x8 = vit_backbone[2][8](x7)
        x9 = vit_backbone[2][9](x8)
        x10 = vit_backbone[2][10](x9)
        x11 = vit_backbone[2][11](x10)
        x = vit_backbone[3](x11)
        x = vit_backbone[4](x)
        
        return x, x3, x6, x9, x11

class SetrMLA(nn.Module):

    def __init__(self, vit_backbone, num_class, bilinear = False):
        super(SetrMLA, self).__init__()
        self.embedding_dim = 768
        self.vit_backbone = VitBackbone(vit_backbone)
        #self.channel_reduction = nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=3, padding=1)
        self.num_classes = num_class
        self._init_decode()

    def _init_decode(self):
        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        # fmt: off
        self.output_net = IntermediateSequential(return_intermediate=False)
        self.output_net.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim, out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        )
        self.output_net.add_module(
            "upsample_1",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        # fmt: on


    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(
                self.embedding_dim, int(self.embedding_dim / 2), 1, 1,
                padding=self._get_padding('VALID', (1, 1),),
            ),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_intmd, model_out

    def forward(self, x):   
        bs = x.size(0)     
        score = self.vit_backbone(x)
        #print(score.size())
        x3 = score[1]
        x6 = score[2]
        x9 = score[3]
        x11 = score[4]
        x3 = torch.reshape(x3, (bs, 24, 24, 768))
        x6 = torch.reshape(x6, (bs, 24, 24, 768))
        x9 = torch.reshape(x9, (bs, 24, 24, 768))
        x11 = torch.reshape(x11, (bs, 24, 24, 768))
        #print(score.size())
        x3 = x3.view(x3.size(0), 24, 24, 768)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x6 = x6.view(x6.size(0), 24, 24, 768)
        x6 = x6.permute(0, 3, 1, 2).contiguous()
        x9 = x9.view(x9.size(0), 24, 24, 768)
        x9 = x9.permute(0, 3, 1, 2).contiguous()
        x11 = x11.view(x11.size(0), 24, 24, 768)
        x11 = x11.permute(0, 3, 1, 2).contiguous()

        temp_x = x3
        key0_intmd_in = self.net1_in(temp_x)
        key0_out = self.net1_out(key0_intmd_in)

        temp_x = x6
        key1_in = self.net2_in(temp_x)
        key1_intmd_in = key1_in + key0_intmd_in
        key1_intmd_out = self.net2_intmd(key1_intmd_in)
        key1_out = self.net2_out(key1_intmd_out)

        temp_x = x9
        key2_in = self.net3_in(temp_x)
        key2_intmd_in = key2_in + key1_intmd_in
        key2_intmd_out = self.net3_intmd(key2_intmd_in)
        key2_out = self.net3_out(key2_intmd_out)

        temp_x = x11
        key3_in = self.net4_in(temp_x)
        key3_intmd_in = key3_in + key2_intmd_in
        key3_intmd_out = self.net4_intmd(key3_intmd_in)
        key3_out = self.net4_out(key3_intmd_out)

        out = torch.cat((key0_out, key1_out, key2_out, key3_out), dim=1)
        out = self.output_net(out)
        return out
    
 

vit = timm.create_model('vit_base_patch16_384', pretrained=True)
vit_backbone = nn.Sequential(*list(vit.children())[:5])
setrMLA = SetrMLA(vit_backbone,5)
pred = setrMLA(torch.rand([1,3,384,384]))
print(pred.size())

