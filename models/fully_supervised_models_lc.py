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
import TransFCN
import multi_res_vit
import resvit_timm
import numpy as np

#import fcn8s
import fcn16s
import fcn
import line_profiler

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
    parser.add_argument('--multi_lr', type=bool, default=False,help="If true, uses different lr for backbone and decoder")
    parser.add_argument('--scheduler', type=U.str2bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--iter_every', default=1, type=int,help="Accumulate compute graph for iter_size step")
    parser.add_argument('--benchmark', default=False, type=U.str2bool, help="enable or disable backends.cudnn")

    #Transformer parameters
    parser.add_argument('--depth', type=int, default=1, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads in a block')
    parser.add_argument('--dim', type=int, default=768, help='Dimension to which patches are projected')
    parser.add_argument('--mlp_dim', type=int, default=3072, help='Hidden dimension in feed forward layer')
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
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = cs.Cityscapes(root=args.dataroot_cs,
                               split='train', transform=train_transform)
        val_dataset = cs.Cityscapes(root=args.dataroot_cs,
                             split='val', transform=val_transform)
        num_classes= 19
        loss_weights = np.load('/users/a/araujofj/loss_weights.npy')
        loss_weights = loss_weights/np.min(loss_weights) + 0.1
        loss_weights = torch.from_numpy(loss_weights).float().to('cuda')
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

    
    # ------------
    # model
    # ------------
    print("chosen model:")
    print(args.model.upper())
    if args.model.upper()=='RESVIT':
        resnet50 = models.resnet50(pretrained=args.pretrained)
        resnet50_backbone = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
        model = resnet50ViT.ResViT(pretrained_net=resnet50_backbone, num_class=num_classes, dim=args.dim, depth=args.depth, heads=args.num_heads, mlp_dim=args.mlp_dim, batch_size = args.batch_size, trans_img_size=args.size_img//8, feat = "feat2")
        print("created resvit model")
        #model = fcn16s.FCN16s(n_class= num_classes)
        #model = models.segmentation.fcn_resnet101(pretrained=args.pretrained,num_classes=num_classes)

    elif args.model.upper()=='RESVIT_TIMM':
        vit = timm.models.vit_base_r50_s16_384(pretrained=True)
        resvit_timm_backbone = nn.Sequential(*list(vit.children())[:-1])
        model = resvit_timm.ResViT_timm(resvit_timm_backbone, num_class=num_classes)
        print("created pre-trained hybrid vit model")

    elif args.model.upper()=='MULTIRESVIT':
        resnet50 = models.resnet50(pretrained=args.pretrained)
        resnet50_backbone = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
        model = multi_res_vit.MultiResViT(pretrained_net=resnet50_backbone, num_class=num_classes, dim=args.dim, depth=args.depth, heads=args.num_heads, mlp_dim=args.mlp_dim)
    
    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained)
        if args.pretrained:
            model.classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
    
    elif args.model.upper()=='SETR':
        vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        vit_backbone = nn.Sequential(*list(vit.children())[:5])
        model = setr.Setr(num_class=num_classes, vit_backbone=vit_backbone, bilinear = False)
    
    elif args.model.upper()=='TRANSFCN':
        FCN = models.segmentation.fcn_resnet50(pretrained=args.pretrained, progress=True, aux_loss=None)
        backbone = nn.Sequential(*list(FCN.children())[:1])
        transformer = vit.ViT(
            image_size = 64,
            patch_size = 1,
            num_classes = 64, #not used
            dim = args.dim,
            depth = args.depth,    #number of encoders
            heads = args.num_heads,    #number of heads in self attention
            mlp_dim = args.mlp_dim,   #hidden dimension in feedforward layer
            channels = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        model = TransFCN.TransFCN(backbone, transformer, num_classes)
    
    elif args.model.upper()=='FCN':
        model = models.segmentation.fcn_resnet50(pretrained=args.pretrained)
        if args.pretrained:
            model.classifier[4] = nn.Conv2d(512, num_classes, 1, 1)
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1, 1)
    else:
        raise Exception('model must be "FCN", "DLV3", "RESVIT', "VIT (all upper case)")
    #model.to(device)

    
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
    
    if args.version == 0:
        loss_weights = torch.tensor([1.1, 78.45, 2.11, 10.37])
        #loss_weights = torch.tensor([5.42373264, 46.43293672, 1.64619769, 50.49834979])
    else:
        loss_weights = torch.tensor([1.1, 63.60, 2.16, 10.56, 43.98])

    loss_weights.to(device)
    if args.landcover:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_weights).to(device), ignore_index=255) # On ignore la classe border.
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    #torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)


    if not args.mixed_precision:
        ev.train_fully_supervised(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,\
            criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=args.scheduler,auto_lr=args.auto_lr,\
                model_name=args.model_name,benchmark=args.benchmark, save_best=args.save_best,save_all_ep=args.save_all_ep,\
                    device=device,num_classes=num_classes)
    else:
        mp.mixed_precision_train(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,criterion=criterion,\
            optimizer=optimizer,scheduler=args.scheduler,auto_lr=args.auto_lr, save_folder=save_dir,model_name=args.model_name,benchmark=False,save_all_ep=True,\
                 save_best=False, save_val_results = False, device=device,num_classes=num_classes)



if __name__ == '__main__':
    main()