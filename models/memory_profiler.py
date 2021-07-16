import numpy as np 
import torch
import torch.nn as nn
from torchvision import models
import resvit_small
import fcn_small
import sys
sys.path.insert(1, '../utils')
from memoryUtils import Hook
import memoryUtils as atlasUtils
import json
import os

def maxmem():
    return torch.cuda.max_memory_allocated()

def curmem():
    return torch.cuda.memory_allocated()

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0
    return size

def apply_hook(net):
    hookF = []
    hookB = []
    for i,layer in enumerate(list(net.children())):
        if not isinstance(layer,torch.nn.ReLU) and not isinstance(layer,torch.nn.LeakyReLU):
            print('Hooked to {}'.format(layer))
            hookF.append(Hook(layer))
            hookB.append(Hook(layer,backward=True))
    print('hook len :', len(hookF), len(hookB))
    return hookF, hookB

def rev_apply_hook(net):
    hookF = []
    hookB = []
    for name, layer in net.named_modules():
        #print(name, layer, type(layer))
        if name != "" and (isinstance(layer, torch.nn.modules.upsampling.Upsample) or isinstance(layer, torch.nn.Conv3d) or isinstance(layer, torch.nn.MaxPool3d)):
            print('Hooked to {}'.format(name))
            hookF.append(Hook(layer))
            hookB.append(Hook(layer,backward=True))
    print('hook len :', len(hookF), len(hookB))
    return hookF, hookB
    
def main():
    heads = 4
    gpu = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda")
    memory_callback = {}
    inchan = 1
    chanscale = 1
    chans = [i//chanscale for i in [64, 128, 256, 512, 1024]]
    num_classes = 5
    interp = None

    resnet50_dilation = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
    backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
    
    #mod = resvit_small.Resvit(backbone=backbone_dilation, num_class=5, dim=768, depth=1, heads=heads, mlp_dim=3072, ff=True)

    mod = fcn_small.FCN_(backbone_dilation, num_class=num_classes, dim=768)

    mod.to(device)
    hookF, hookB = rev_apply_hook(mod)
    # hookF, hookB = apply_hook(mod)
    memory_callback['model'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    fact = 0.5
    # s = (80,80,32)
    # s = (112,112,48)
    # s = (160,160,64)
    # s = (256,256,112)
    s = (512,512,208)
    # x = torch.from_numpy(np.random.rand(1,1,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()
    x = torch.rand([1,3,512,512]).float()
    x = x.to(device)
    memory_callback['input'] = {'max' : convert_bytes( maxmem()), 'cur' : curmem()}
    y = torch.rand([1,512,512]).long()
    y = y.to(device)
    memory_callback['output'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    opti = torch.optim.SGD(mod.parameters(), lr=0.01)
    opti.zero_grad()
    loss = nn.CrossEntropyLoss(ignore_index=255)
    mod.train()
    out = mod(x)
    del x
    memory_callback['forward'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    l = loss(out, y)
    del y, out
    memory_callback['loss'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    l.backward()
    memory_callback['backward'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    opti.step()
    memory_callback['step'] = {'max' : convert_bytes( maxmem()), 'cur' : convert_bytes( curmem())}
    memory_callback['hookF'] = []
    memory_callback['hookB'] = []
    for i,j in zip(hookF, hookB):
        memory_callback['hookF'].append({'max' : i.max_mem, 'cur' : i.cur_mem, 'name':str(i.name)})
        memory_callback['hookB'].append({'max' : j.max_mem, 'cur' : j.cur_mem, 'name':str(j.name)})
    print("LEN : ", len(memory_callback['hookF']), len(memory_callback['hookF']))
    with open('callback_memoryFCN.json', 'w') as f:
      json.dump(memory_callback, f, indent=4)
    #with open('callback_'+str(heads)+'heads'+'.json', 'w') as f:
    #    json.dump(memory_callback, f, indent=4)

if __name__ == '__main__':
    main()