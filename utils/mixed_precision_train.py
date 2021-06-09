import torch 
import torch.nn as nn
import numpy as np
import utils as U
import random 
import get_datasets as gd
from matplotlib import colors
import os
from torch_lr_finder import LRFinder
import matplotlib as plt
from PIL import Image
import sys
sys.path.insert(1, '../metrics')
import stream_metrics as sm
import line_profiler


# -----------------------------------------------------
# Trains model using mixed precision with pure Pytorch
# TODO: incorporate Pytorch Lightning
# Edited version of code taken from https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py
# -----------------------------------------------------

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

#@profile
def mixed_precision_train(model,n_epochs,train_loader,val_loader,criterion,optimizer,scheduler,auto_lr,\
        save_folder,model_name,benchmark=False,save_all_ep=True, save_best=False, save_val_results = False, device='cpu',num_classes=21):
    """
        A complete training of fully supervised model. 
        save_folder : Path to save the model, the courb of losses,metric...
        benchmark : enable or disable backends.cudnn 
        save_all_ep : if True, the model is saved at each epoch in save_folder
        scheduler : if True, the model will apply a lr scheduler during training
        auto_lr : Auto lr finder 
    """

    #set metrics
    metrics = sm.StreamSegMetrics(num_classes)

    torch.backends.cudnn.benchmark=benchmark
    
    if auto_lr:
        print('Auto finder for the Learning rate')
        lr_finder = LRFinder(model, optimizer, criterion,memory_cache=False,cache_dir='/tmp', device=device)
        lr_finder.range_test(train_loader,start_lr=10e-5, end_lr=10, num_iter=100)
    
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader) * n_epochs)) ** 0.9)

    #define scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    #useful variables
    best_score = 0.0
    cur_itrs = 0
    cur_itrs = 0
    interval_loss = 0
    train_losses = []
    epoch_losses = []
    iou_train = []
    iou_test = []
    accuracy_train = []
    accuracy_test = []
    model.to(device)

    #train loop
    for epoch in range(n_epochs):
        print("in epoch loop : ", epoch)
        epoch_loss = 0
        cur_itrs = 0
        
        #--------- train step ---------------
        for (img, mask) in train_loader:
            cur_itrs += 1

            img = img.to(device)
            mask = mask.to(device)

            # print("data loaded : ", i)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(img)
                try:
                    outputs = outputs['out']
                except:
                    pass
                loss = criterion(outputs, mask)
            
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            
            if scheduler:
                lr_scheduler.step()
            
            if cur_itrs%200 == 0:
                print(epoch, cur_itrs, loss.item()) 
        
        #--------- validation step ---------------
        print("validation...")
        model.eval()
        val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics, save_val_results=save_val_results)
        print(metrics.to_str(val_score))
        
        if val_score['Mean IoU'] > best_score:
            best_score = val_score['Mean IoU']
            #save ckpt
            save_model = model_name+'.pt'
            save = os.path.join(save_folder,save_model)
            #torch.save({
            #"cur_itrs": cur_itrs,
            #"model_state": model.state_dict(),
            #"best_score": best_score,
            #}, save)
            torch.save(model,save)

            print("Model saved in %s" % save_folder)
        #back to train mode
        model.train()
        
        epoch_losses.append(epoch_loss/len(train_loader))
    
    #end for epochs
    U.save_curves(path=save_folder,loss_train=train_losses, epoch_losses=epoch_losses)
