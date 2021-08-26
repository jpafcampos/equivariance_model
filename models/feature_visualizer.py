import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms

#model = models.resnet50(pretrained=True)
model = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])

#print(model)
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

# visualize the first conv layer filters
#plt.figure(figsize=(20, 17))
#for i, filter in enumerate(model_weights[0]):
#    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
#    plt.imshow(filter[0, :, :].detach(), cmap='gray')
#    plt.axis('off')
#    plt.savefig('../outputs2/filter.png')
#plt.show()


# read and visualize an image
img = cv.imread(f"../N-33-60-D-c-4-2_24.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
#for num_layer in range(len(outputs)):
#    plt.figure(figsize=(30, 30))
#    layer_viz = outputs[num_layer][0, :, :, :]
#    layer_viz = layer_viz.data
#    print(layer_viz.size())
#    for i, filter in enumerate(layer_viz):
#        if i == 64: # we will visualize only 8x8 blocks from each layer
#            break
#        plt.subplot(8, 8, i + 1)
#        plt.imshow(filter, cmap='gray')
#        plt.axis("off")
#    print(f"Saving layer {num_layer} feature maps...")
#    plt.savefig(f"../outputs2/layer_{num_layer}.png")
#    # plt.show()
#    plt.close()

#img = torch.from_numpy(img)
img.detach()
img.numpy()
model = models._utils.IntermediateLayerGetter(model, {'layer1': 'feat1', 'layer2': 'feat2','layer3': 'feat3','layer4': 'feat4'})
pred = model(img)
for i in range(4):
    pred_i = pred["feat"+str(i+1)]
    print(pred_i.size())
    pred_i = torch.squeeze(pred_i)
    plt.imshow(pred_i.detach()[1])
    plt.savefig(f"../outputs2/pred_feat_ch1"+str(i+1)+".png")
