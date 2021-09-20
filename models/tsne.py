import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms
import resvit_small
import setr
import fcn_small
import timm
import os
import PIL
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm

model_name = "setr"

colors_per_class = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [25, 60, 243],
    4 : [156, 72, 132]
}
colors_per_class = {
    'background' : 'orange',
    'buildings' : 'maroon',
    'woods' : 'green',
    'water' : 'blue',
    'roads' : 'yellow'
}

classes = ['background', 'buildings', 'woods', 'water', 'roads']
classes_id = {
    'background' : 0,
    'buildings'  : 1,
    'woods'      : 2,
    'water'      : 3,
    'roads'      : 4
}

def to_tensor_target_lc(mask):
    # For the landcoverdataset
    mask = np.array(mask)
    mask = np.mean(mask, axis=2) 

    return torch.LongTensor(mask)

gpu = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#device = torch.device("cuda")
device = torch.device("cpu")
num_classes = 5
H = 24
dim = 768
depth = 1
num_heads = 2
mlp_dim = 3072


downsample = transforms.Resize((64,64))
upsample = transforms.Resize((64,64))

img = cv.imread(f"../N-33-60-D-c-4-2_24.jpg")
gt = Image.open(f"../N-33-60-D-c-4-2_24_m.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gt = downsample(gt)
gt = to_tensor_target_lc(gt)
gt = gt.reshape(4096)
print(gt.size())
unique, count = np.unique(gt, return_counts=True)
print(unique, count)


# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
])

img = np.array(img)
# apply the transforms
img = transform(img)

# unsqueeze to add a batch dimension
img = img.unsqueeze(0)

# PREPARE MODEL
if model_name == "resvit":
    resnet50_dilation = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
    model = resvit_small.Resvit(backbone=backbone_dilation, num_class=num_classes, dim=dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

if model_name == "fcn":
    resnet50_dilation = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
    backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
    model = fcn_small.FCN_(backbone_dilation, num_class=num_classes, dim=768)

if model_name == "setr":
    vit = timm.create_model('vit_base_patch16_384', pretrained=True)
    vit_backbone = nn.Sequential(*list(vit.children())[:5])
    model = setr.Setr(num_class=num_classes, vit_backbone=vit_backbone, bilinear = False)


model_root = "/users/a/araujofj/data/save_model/resvit/69/resvit_dilation.tar" #cyclic lr
model_root = "/users/a/araujofj/fcn_baseline_lc1.pt" #best FCN
model_root = "/users/a/araujofj/data/save_model/setr/7/setr.tar"

checkpoint = torch.load(model_root, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
print("loaded state dict")



# !!! DONT FORGET TO SET MODEL TO EVAL MODE !!!
model.eval()

y, features = model(img)

features = features.squeeze(0)  #C x H/8 x W/8
features = upsample(features)
features = features.reshape(768, 4096)
features = features.permute(1,0).contiguous()
features = features.detach().numpy()
print(features.shape)



tsne = TSNE(2, perplexity=45, learning_rate=100, n_iter=4000,verbose=1, init='pca')
tsne_proj = tsne.fit_transform(features)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20b')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 5
for lab in classes[1:]:
    print(lab)
    indices = gt==classes_id[lab]
    #ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=colors_per_class[lab], label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()

plt.savefig('tsne.png')

