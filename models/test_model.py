import torch
from torch.nn.modules import loss
from torchvision import models
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import timm
import sys
sys.path.insert(1, '../utils')
sys.path.insert(1, '../datasets')
import my_datasets as mdset
import cityscapes as cs
import utils as U
import ext_transforms as et
import coco_utils as cu
import eval_train as ev
import mixed_precision_train as mp
from argparse import ArgumentParser
import torch.utils.data as tud
import resnet50ViT
import setr
import vit
import multi_res_vit
import resvit_timm
import my_fcn
import resvit_dilation
import numpy as np
import sys
sys.path.insert(1, '../metrics')
import stream_metrics as sm

#import fcn8s
import fcn16s
import fcn
import line_profiler

model_name = "fcn"
gpu = 1
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print("device used:",device)

num_classes = 5
dataroot_landcover = "/local/DEEPLEARNING/landcover_v1"

print('Loading Landscape Dataset')
test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test")
print('Success load Landscape Dataset')

test_loader = torch.utils.data.DataLoader(test_dataset,num_workers=4,batch_size=6)

if model_name == "fcn":
    #model = models.segmentation.fcn_resnet50(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, num_classes, 1, 1)
    #model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
    resnet50_dilation = models.resnet50(False, replace_stride_with_dilation=[False, True, True])
    backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
    model = my_fcn.FCN_(backbone_dilation, num_class=num_classes)

elif model_name == "resvit":
    resnet50_dilation = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
    backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
    model = resvit_dilation.Resvit(backbone_dilation, num_class=num_classes, heads=1)


model_root = "/users/a/araujofj/data/save_model/FCN/15/my_fcn.tar"

#model_root = "/users/a/araujofj/data/save_model/resvit/16/resvit_dilation.tar"
checkpoint = torch.load(model_root)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# !!! DONT FORGET TO SET MODEL TO EVAL MODE !!!
model.eval()


def validate(model, loader, device, metrics, save_val_results = False):
    """Do validation"""
    metrics.reset()

    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = U.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            try:
                outputs = outputs['out']
            except:
                pass
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(plt.ticker.NullLocator())
                    ax.yaxis.set_major_locator(plt.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score

metrics = sm.StreamSegMetrics(num_classes)
val_score = validate(model=model, loader=test_loader, device=device, metrics=metrics, save_val_results=False)
print(metrics.to_str(val_score))