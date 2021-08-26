from functools import wraps
import torch
from torch import nn

from vit import Attention

import resvit_small
from torchvision import models


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, model, device = None):
        super().__init__()
        self.model = model

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.model.transformer.transformer, Attention)   ##maybe model.vit.transformer
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.model

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.model(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        attns = torch.stack(recordings, dim = 1)
        return pred, attns

#print("ok")
#resnet50_dilation = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
#backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
#model = resvit_small.Resvit(backbone=backbone_dilation, num_class=5, dim=768, depth=1, heads=1, mlp_dim=1024, ff=True)
##print(model)
#model = Recorder(model)
#img = torch.randn(1, 3, 512, 512)
#preds, attns = model(img)
#print(attns.size()) #[1,1,1,4096,4096]