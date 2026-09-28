import torch
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable  

import pickle
import json 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pdb

from utils.config import MyTracker, EarlyStopping

from torch.optim import lr_scheduler  

def train(network, dataset, config, data_mode, epoch_start ):
 
    dev        = "cuda:" + str(config["Training"]["gpu_id"])
    device     = torch.device(dev if torch.cuda.is_available() else "cpu") 
    eval_every = config["Training"]["eval_every"]
 
    num_images = 0

    train_loss = []  
    train_psnr = []   
    train_disc_loss = [] 

    print("Constructing training and validate Set !!! ")  
       
 
    early_stopping = EarlyStopping(patience = config["Training"]["patience"], delta = 1e-4)
    scheduler      = lr_scheduler.ReduceLROnPlateau(optimizer = network.optimizer, mode='min', factor=0.5, patience=5)
    
    tracker = MyTracker(config["stats_path"])   
 
        
    for epoch in range(epoch_start, config["Training"]["epochs"]):
        print('Starting Epoch...', epoch)  
        train_output = network.train(dataset, config, epoch) 

        get_last_lr = scheduler.get_last_lr()[0] 

        if "disc_loss_per_epoch" in train_output.keys():

            train_psnr_per_epoch = train_output["psnr_per_epoch"]   
            train_loss_per_epoch = train_output["loss_per_epoch"]
            train_disc_loss_per_epoch  = train_output["disc_loss_per_epoch"]
            tracker.train_stats.log({"EP": epoch, "PSNR": train_psnr_per_epoch, "Loss": train_loss_per_epoch, "DISC_LOSS": train_disc_loss_per_epoch, "LR": get_last_lr})

        else:

            train_psnr_per_epoch = train_output["psnr_per_epoch"] 
            train_loss_per_epoch = train_output["loss_per_epoch"]  
            tracker.train_stats.log({"EP": epoch, "PSNR": train_psnr_per_epoch, "Loss": train_loss_per_epoch, "LR": get_last_lr})

        train_loss.append(train_loss_per_epoch)  
        train_psnr.append(torch.tensor(train_psnr_per_epoch).mean().item())     
 

        if (epoch+1) % eval_every == 0: 
            
            valid_output = network.evaluation(dataset, config, epoch)   
          
            valid_loss_per_epoch = valid_output["loss_per_epoch"] 
            valid_images_list = valid_output["samples_list"]

            valid_log = {**{k: v for k, v in valid_output.items() if k != "samples_list"}, "EP": epoch, "LR": get_last_lr}
            
            tracker.valid_stats.log(valid_log)


            # Put logged data into csv
            tracker.train_stats.log_to_csv(epoch)
            tracker.valid_stats.log_to_csv(epoch) 

            for index_list_, valid_images_list_ in enumerate(valid_images_list):
                tracker.image_list(valid_images_list[index_list_]["ground_truth"], epoch = epoch, image_key= "%d-GT"   % index_list_)
                tracker.image_list(valid_images_list[index_list_]["predictions"],  epoch = epoch, image_key= "%d-Pred" % index_list_)
 
            if hasattr(network, 'generat_scheduler'):    
                network.generat_scheduler.step(valid_loss_per_epoch)

            if hasattr(network, 'discrim_scheduler'):  
                network.discrim_scheduler.step(valid_loss_per_epoch)
            else:
                scheduler.step(valid_loss_per_epoch)   
            
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))   
            print('Saved model iteration' +  str(epoch) )
            early_stopping(network, config["ckpt_path"], epoch, valid_loss_per_epoch, valid_output["psnr_per_epoch"])


        if early_stopping.early_stop:
            print("Early stopping")  
            break

    return scheduler.get_last_lr()[0]