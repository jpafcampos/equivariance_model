from matplotlib import pyplot as plt
import os
from os.path import isfile,join
import numpy as np

path = "/users/a/araujofj/data/save_model/resvit/75/"
#path = "/users/a/araujofj/data/save_model/FCN/34"
def plot_loss_metrics(folder,model_name):
    l_npy = [f for f in os.listdir(folder) if isfile(join(folder,f)) and f.endswith(".npy")] # Load all numpy file in the best folder
    for f in l_npy:
        curv = np.load(join(folder,f))
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.title(model_name.upper())
        plt.plot(curv)
        plt.xlabel("iterations")
        plt.ylabel(f.upper())
        plt.savefig('./'+f+'.png')


plot_loss_metrics(path,"hybrid vit, Cityscapes")

#plt.figure(figsize=(20,10))
#plt.plot(train_loss_history)
#plt.xlabel("Time")
#plt.ylabel("Loss")
#plt.title("Loss function evolution")
#plt.savefig(save_folder+'_loss_train.png')