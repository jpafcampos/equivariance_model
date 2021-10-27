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
sys.path.insert(1, '../metrics')
import stream_metrics as sm
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
import resvit_small
import setr
import vit
import rot_eq_fcnVit
import segformer
import fcn_volo
import transFCN
import multi_res_vit
import numpy as np

import fcn_small
from functools import partial
import my_fcn
import line_profiler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##@profile
def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    # Learning parameters
    parser.add_argument('--auto_lr', type=U.str2bool, default=False,help="Auto lr finder")
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--multi_lr', type=U.str2bool, default=False,help="If true, uses different lr for backbone and decoder")
    parser.add_argument('--scheduler', type=U.str2bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--iter_every', default=1, type=int,help="Accumulate compute graph for iter_size step")
    parser.add_argument('--benchmark', default=False, type=U.str2bool, help="enable or disable backends.cudnn")
    parser.add_argument('--bilinear_up', default=False, type=U.str2bool, help="if True creates original FCN, otherwise replaces the decoder")

    #Transformer parameters
    parser.add_argument('--depth', type=int, default=1, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of heads in a block')
    parser.add_argument('--dim', type=int, default=512, help='Dimension to which patches are projected')
    parser.add_argument('--mlp_dim', type=int, default=1024, help='Hidden dimension in feed forward layer')
    parser.add_argument('--ff', type=U.str2bool, default=True, help='Whether to use feed forward layer in att block')
    # Model and eval

    parser.add_argument('--mixed_precision', default = False, type=bool)
    parser.add_argument('--model', default='FCN', type=str,help="FCN or DLV3 model")
    parser.add_argument('--pretrained', default=True, type=U.str2bool,help="Use pretrained pytorch model")
    parser.add_argument('--eval_angle', default=False, type=U.str2bool,help=\
        "If true, it'll eval the model with different angle input size")
    
    
    # Data augmentation
    parser.add_argument('--rotate', default=False, type=U.str2bool,help="Use random rotation as data augmentation")
    parser.add_argument('--pi_rotate', default=True, type=U.str2bool,help="Use only pi/2 rotation angle")
    parser.add_argument('--p_rotate', default=0.25, type=float,help="Probability of rotating the image during the training")
    parser.add_argument('--scale', default=False, type=U.str2bool,help="Use scale as data augmentation")
    parser.add_argument('--landcover', default=False, type=U.str2bool,\
         help="Use Landcover dataset instead of VOC and COCO")
    parser.add_argument('--version', default=0, type=int, help="landcover data set version")
    parser.add_argument('--lc_new', default = False, type=U.str2bool, help="Uses new data division in LC")
    parser.add_argument('--lc_augs', default = False, type=U.str2bool, help="perform data augs as in the Landcover paper")
    parser.add_argument('--size_img', default=512, type=int,help="Size of input images")
    parser.add_argument('--size_crop', default=480, type=int,help="Size of crop image during training")
    
    # Dataloader and gpu
    parser.add_argument('--nw', default=0, type=int,help="Num workers for the data loader")
    parser.add_argument('--pm', default=True, type=U.str2bool,help="Pin memory for the dataloader")
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    
    # Datasets 
    parser.add_argument('--split', default=False, type=U.str2bool, help="Split the dataset")
    parser.add_argument('--split_ratio', default=0.3, type=float, help="Amount of data we used for training")
    parser.add_argument('--dataroot_voc', default='/data/voc2012', type=str)
    parser.add_argument('--dataroot_sbd', default='/data/sbd', type=str)
    #parser.add_argument('--dataroot_landcover', default='/local/DEEPLEARNING/landcover_v1', type=str)
    parser.add_argument('--dataroot_cs', default='/local/DEEPLEARNING/cityscapes', type=str)
    parser.add_argument('--class_weights', default=False, type=U.str2bool)

    
    # Save parameters
    parser.add_argument('--model_name', type=str,help="what name to use for saving")
    parser.add_argument('--save_dir', default='/data/save_model', type=str)
    parser.add_argument('--save_all_ep', default=False, type=U.str2bool,help=\
        "If true it'll save the model every epoch in save_dir")
    parser.add_argument('--save_best', default=False, type=U.str2bool,help="If true will only save the best epoch model")
    args = parser.parse_args()
    
    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    
    # ------------
    # data
    # ------------

    if args.version == 0:
        dataroot_landcover = "/local/DEEPLEARNING/landcover"
    else:
        dataroot_landcover = "/local/DEEPLEARNING/landcover_v1"

    if args.lc_new:
        dataroot_landcover = "/local/DEEPLEARNING/lc_new"

    if args.size_img < args.size_crop:
        raise Exception('Cannot have size of input images less than size of crop')
    size_img = (args.size_img,args.size_img)
    size_crop = (args.size_crop,args.size_crop)

    if not args.landcover:
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(args.size_crop, args.size_crop)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(args.size_crop, args.size_crop)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = cs.Cityscapes(root=args.dataroot_cs,
                               split='train', transform=train_transform)
        val_dataset = cs.Cityscapes(root=args.dataroot_cs,
                             split='val', transform=val_transform)
        test_dataset = cs.Cityscapes(root=args.dataroot_cs, split='test', transform=val_transform)
        num_classes= 19


    else:
        print('Loading Landscape Dataset')
        if args.version == 1:
            num_classes = 5
        else: 
            num_classes = 4
        train_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="train",\
            rotate=args.rotate,pi_rotate=args.pi_rotate,p_rotate=args.p_rotate,size_img=size_img,size_crop=size_crop)
        val_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="val", size_img=size_img, size_crop=size_crop)
        test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test")
        print('Success load Landscape Dataset')

    # Print len datasets
    print("There is",len(train_dataset),"images for training and",len(val_dataset),"for validation")
    print("the number of classes is", num_classes)
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)#,collate_fn=U.my_collate)
    dataloader_val = torch.utils.data.DataLoader(val_dataset,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)
    dataloader_test = torch.utils.data.DataLoader(test_dataset,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)

    
    # ------------
    # model
    # ------------
    print("chosen model:")
    print(args.model.upper())
    if args.model.upper()=='RESVIT_SMALL':
        print("Pretrained backbone:", args.pretrained)
        resnet50_dilation = models.resnet50(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
        backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
        model = resvit_small.Resvit(backbone=backbone_dilation, num_class=num_classes, dim=args.dim, depth=args.depth, heads=args.num_heads, mlp_dim=args.mlp_dim, ff=args.ff)
        print("created resvit small model")
        print("Dim, depth, heads and MLP dim: ", args.dim, args.depth, args.num_heads, args.mlp_dim)
        print("Feed Forward:", args.ff)

    elif args.model.upper()=='RESVIT':
        print("Pretrained backbone:", args.pretrained)
        resnet50_dilation = models.resnet50(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
        backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
        model = resnet50ViT.Resvit(backbone_dilation, num_class=num_classes, heads=args.num_heads, mlp_dim=args.mlp_dim)
        print("created resvit with resnet50 backbone replacing stride with dilation")
        print("Dim, depth, heads and MLP dim: ", args.dim, args.depth, args.num_heads, args.mlp_dim)
    
    elif args.model.upper()=='ROT_EQ_RESVIT':
        print("rotation equivariant resvit model")
        model = rot_eq_fcnVit.create_model_groupy(group_elements=4, classes=num_classes, dim=args.dim, depth=args.depth, heads=args.num_heads, mlp_dim=args.mlp_dim)
        print("Dim, depth, heads and MLP dim: ", args.dim, args.depth, args.num_heads, args.mlp_dim)

    elif args.model.upper()=='FCN_VOLO':
        print("Pretrained backbone:", args.pretrained)
        resnet50_dilation = models.resnet50(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
        backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
        model = fcn_volo.FCN_volo(backbone_dilation, num_class=num_classes, dim=args.dim)
        print("created FCN + VOLO with resnet50 backbone replacing stride with dilation")
        print("Dim, depth, heads and MLP dim: ", args.dim, args.depth, args.num_heads, args.mlp_dim)

    elif args.model.upper()=='SEGFORMER':
        model = segformer.Segformer(
            pretrained="/users/a/araujofj/weights/mit_b3.pth",
            img_size=512,
            patch_size=4,
            num_classes=5,
            embed_dims=[64,128,320,512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3,4,18,3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1,
            decoder_dim = 768   
        )


    elif args.model.upper()=='MULTIRESVIT':
        print("Pretrained backbone:", args.pretrained)
        resnet50_dilation = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
        model = multi_res_vit.MultiResViT(backbone_dilation, num_class=num_classes, dim=args.dim, depth=args.depth, heads=args.num_heads, mlp_dim=args.mlp_dim)
        print("created multires hybrid bit")

    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained)
        if args.pretrained:
            model.classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
    
    elif args.model.upper() == 'SETR':
        vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        vit_backbone = nn.Sequential(*list(vit.children())[:5])
        model = setr.Setr(num_class=num_classes, vit_backbone=vit_backbone, bilinear = False)
        print("created SETR model")

    elif args.model.upper()=='FCN':
        if args.bilinear_up:
            model = models.segmentation.fcn_resnet50(pretrained=args.pretrained)
            if args.pretrained:
                model.classifier[4] = nn.Conv2d(512, num_classes, 1, 1)
                model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
            print("Pretrained backbone:", args.pretrained)
        else:
            resnet50_dilation = models.resnet50(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
            backbone_dilation = models._utils.IntermediateLayerGetter(resnet50_dilation, {'layer4': 'feat4'})
            model = fcn_small.FCN_(backbone_dilation, num_class=num_classes, dim=args.dim)
            #model = my_fcn.FCN_(backbone_dilation, num_class=num_classes)
            print("created small FCN model")
    else:
        raise Exception('model not found')
    print("number of params")
    print(count_parameters(model))
    #model.to(device)

    #while(1):
    #    a = 1

    
    # ------------
    # save
    # ------------
    save_dir = U.create_save_directory(args.save_dir)
    print('model will be saved in',save_dir)
    U.save_hparams(args,save_dir)

    # ------------
    # training
    # ------------
    # Auto lr finding
    
    print(save_dir)
    print("class weights ", args.class_weights)
    if args.class_weights:
        print("creating class weights")
        if args.version == 0:
            #loss_weights = torch.tensor([1.1, 78.45, 2.11, 10.37])
            loss_weights = torch.tensor([5.42373264, 46.43293672, 1.64619769, 50.49834979])
        else:
            loss_weights = torch.tensor([1.1, 53.71, 1.84, 9.00, 37.20])

        loss_weights.to(device)
        if args.landcover:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_weights).to(device), ignore_index=255) # On ignore la classe border.
            print("using CrossEntropyLoss with class balancing")
        else:
            loss_weights = torch.tensor([ 1.1,          6.16178822,   1.71923575,  56.39632727,
                             42.14954318,  30.17444113, 177.77029189,  67.05827838,   2.42050082,
                             31.9604042,   9.34098606,  30.34813821, 272.95300993,   5.37351881,
                            138.03022215, 156.89769638, 158.50789819, 374.17642732,  89.20747291])
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_weights).to(device), ignore_index=255)
        #torch.autograd.set_detect_anomaly(True)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        print("using CrossEntropyLoss without weights")

    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)
    #optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.wd)


    if not args.mixed_precision:
        ev.train_fully_supervised(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,\
            criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=args.scheduler,auto_lr=args.auto_lr,\
                model_name=args.model_name,benchmark=args.benchmark, save_best=args.save_best,save_all_ep=args.save_all_ep,\
                    device=device,num_classes=num_classes)
    else:
        mp.mixed_precision_train(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,test_loader=dataloader_test,criterion=criterion,\
            optimizer=optimizer,lr=args.learning_rate, scheduler=args.scheduler,auto_lr=args.auto_lr, save_folder=save_dir,model_name=args.model_name,benchmark=False,save_all_ep=True,\
                 save_best=False, save_val_results = False, device=device,num_classes=num_classes)



if __name__ == '__main__':
    main()