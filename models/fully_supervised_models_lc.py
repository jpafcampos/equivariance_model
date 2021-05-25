import torch
from torchvision import models
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F 
import sys
sys.path.insert(1, '../utils')
sys.path.insert(1, '../datasets')
import my_datasets as mdset
import utils as U
import coco_utils as cu
import eval_train as ev
from argparse import ArgumentParser
import torch.utils.data as tud
import trans_fcn
import resnet50ViT

#import fcn8s
import fcn16s
#import fcn32s
import vgg
import fcn


def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    # Learning parameters
    parser.add_argument('--auto_lr', type=U.str2bool, default=False,help="Auto lr finder")
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--scheduler', type=U.str2bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--iter_every', default=1, type=int,help="Accumulate compute graph for iter_size step")
    parser.add_argument('--benchmark', default=False, type=U.str2bool, help="enable or disable backends.cudnn")
    
    # Model and eval
    parser.add_argument('--model', default='FCN', type=str,help="FCN or DLV3 model")
    parser.add_argument('--pretrained', default=False, type=U.str2bool,help="Use pretrained pytorch model")
    parser.add_argument('--eval_angle', default=True, type=U.str2bool,help=\
        "If true, it'll eval the model with different angle input size")
    
    
    # Data augmentation
    parser.add_argument('--rotate', default=False, type=U.str2bool,help="Use random rotation as data augmentation")
    parser.add_argument('--pi_rotate', default=True, type=U.str2bool,help="Use only pi/2 rotation angle")
    parser.add_argument('--p_rotate', default=0.25, type=float,help="Probability of rotating the image during the training")
    parser.add_argument('--scale', default=True, type=U.str2bool,help="Use scale as data augmentation")
    parser.add_argument('--landcover', default=False, type=U.str2bool,\
         help="Use Landcover dataset instead of VOC and COCO")
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
    parser.add_argument('--dataroot_landcover', default='/share/DEEPLEARNING/datasets/landcover', type=str)
    
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
    if args.size_img < args.size_crop:
        raise Exception('Cannot have size of input images less than size of crop')
    size_img = (args.size_img,args.size_img)
    size_crop = (args.size_crop,args.size_crop)
    if not args.landcover:
        train_dataset_VOC = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='train', \
            download=True,rotate=args.rotate,size_img=size_img,size_crop=size_crop)
        test_dataset = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='val', download=True)
        train_dataset_SBD = mdset.SBDataset(args.dataroot_sbd, image_set='train_noval',mode='segmentation',\
            rotate=args.rotate,size_img=size_img,size_crop=size_crop)
        #COCO dataset 
        if args.extra_coco:
            extra_COCO = cu.get_coco(args.dataroot_coco,'train',rotate=args.rotate,size_img=size_img,size_crop=size_crop)
            # Concatene dataset
            train_dataset = tud.ConcatDataset([train_dataset_VOC,train_dataset_SBD,extra_COCO])
        else:
            train_dataset = tud.ConcatDataset([train_dataset_VOC,train_dataset_SBD])
        num_classes = 21
    else:
        print('Loading Landscape Dataset')
        train_dataset = mdset.LandscapeDataset(args.dataroot_landcover,image_set="trainval",\
            rotate=args.rotate,pi_rotate=args.pi_rotate,p_rotate=args.p_rotate,size_img=size_img,size_crop=size_crop)
        test_dataset = mdset.LandscapeDataset(args.dataroot_landcover,image_set="test")
        print('Success load Landscape Dataset')
        num_classes = 4
    
    split = args.split
    if split==True:
        train_dataset = U.split_dataset(train_dataset,args.split_ratio)
    # Print len datasets
    print("There is",len(train_dataset),"images for training and",len(test_dataset),"for validation")
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)#,collate_fn=U.my_collate)
    dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)

    
    # ------------
    # model
    # ------------
    
    if args.model.upper()=='ResViT':
        resnet50 = models.resnet50(pretrained=True)
        resnet50_backbone = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
        model = resnet50ViT.ResViT(pretrained_net=resnet50_backbone, num_class=num_classes, dim=768, depth=3, heads=6, batch_size = args.batch_size, trans_img_size=32)
        #model = fcn16s.FCN16s(n_class= num_classes)
        #model = models.segmentation.fcn_resnet101(pretrained=args.pretrained,num_classes=num_classes)
    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained,num_classes=num_classes)
    elif args.model.upper()=='ViT':
        pass
    elif args.model.upper()=='TransFCN8s':
        pass
    elif args.model.upper()=='FCN':
        pass
    else:
        raise Exception('model must be "FCN" or "DLV3"')
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

    criterion = nn.CrossEntropyLoss(ignore_index=num_classes) # On ignore la classe border.
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)
    
    ev.train_fully_supervised(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,\
        criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=args.scheduler,auto_lr=args.auto_lr,\
            model_name=args.model_name,benchmark=args.benchmark, save_best=args.save_best,save_all_ep=args.save_all_ep,\
                device=device,num_classes=num_classes)



if __name__ == '__main__':
    main()