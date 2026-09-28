from torch import nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from models.Unet_utils.Unet import  Image2PV 
# from losses.Unet.losses import MSE_Loss, MAE_Loss  

from models.common.unet_regressor_layers import Image2PV
from models.common.unet_regressor_losses import MSE_Loss, MAE_Loss
 
import pdb

from tqdm import tqdm
from torch.autograd import Variable  
import os
import json 
import pandas as pd

import numpy as np
import h5py

def PSNR_torch(original, compressed):  
    B, H, W, P, C = original.permute(0, 3, 4, 1, 2).shape
    mse = torch.sum((original.view(-1) - compressed.view(-1)) ** 2)/(B*P*C*H*W)   
    max_pixel = 255 
    psnr      = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        model_name          = configs["network"] 
        
        self.image_channels = configs["image_channels"]  

        self.seq_length     = configs["seq_length"]
        self.predict_length = configs["predict_length"]

        input_channels      = configs["seq_length"]*configs["image_channels"]
        output_channels     = configs["predict_length"]*configs["image_channels"] 

 
        self.ckpt_path      = configs["ckpt_path"]
        self.stats_path     = configs["stats_path"]
        self.log_folder     = configs["log_folder"]

        dev                 = "cuda:%d" % configs["gpu_id"]
        device              = torch.device(dev if torch.cuda.is_available() else "cpu")  
        self.device         = device
 
        if configs["loss"] == "MSE":
            self.loss_function = MSE_Loss().to(device) 
        elif configs["loss"] == "MAE":
            self.loss_function = MAE_Loss().to(device) 
        else:
            KeyError

        self.model          = Image2PV(configs, device).to(device)  
        self.num_param      = sum(p.numel() for p in self.model.parameters() if p.requires_grad)  

        print("Regressor [%s] Num. params = %d" % (model_name, self.num_param))

        if configs["is_train"]:  
            self.optimizer  = torch.optim.Adam(self.model.parameters(), lr = configs["learning_rate"], betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False)  
        else: 
            self.model      = self.model.eval()
            


    def forward(self, input):
        return self.model(input)  
    
    def _infer_target_frame(self, input):
        y_pred = self.model(input)
        return y_pred


    def inference_target_sequence(self, input_image):  
   
        self.model.eval()  
        with torch.no_grad():
            y_pred = self._infer_target_frame(input_image)   
                
        return y_pred
    
    def update_schedule(self, epoch):
        self.epoch = epoch   
    

    def train(self, trainLoader, epoch):
        pbar = tqdm(trainLoader)  
        self.model.train() 

        trainLossCount = 0 
        num_images = 0 

        trainingLoss_list_epoch = []
        mae_blck = []

        self.update_schedule(epoch) 
        last_iter=len(pbar)-1
        
        for input_index, data in enumerate(pbar):
            idx = data[0];  X_image= data[1] ; X_Irr  = data[2]
            
            X_image = Variable(X_image.float()).to(self.device)   
            X_Irr   = Variable(X_Irr.float()).to(self.device)  
            
            num_images += X_image.size(0) 

            self.optimizer.zero_grad() 

            X_pred = self.model(X_image) 
            loss   = self.loss_function(X_pred, X_Irr.view(-1,1)) 

            loss.backward()
            self.optimizer.step()
    
            mae = torch.mean(torch.abs(X_pred.view(-1,1) - X_Irr.view(-1,1))).detach()
   
            records = {"total_loss": loss.item() , "mae": mae.item(), "EP": epoch, "data_index": input_index}

            mae_blck.append(records["mae"]) 
            trainingLoss_list_epoch.append(records)
            trainLossCount = trainLossCount + records["total_loss"] 


            # if (input_index % last_iter == 0) and input_index > 0:    
            #     dest_file  = '%s/%s/trainloss_epoch_%03d_%05d.txt' % (self.stats_path, self.log_folder, epoch, input_index)
            #     output_file = open(dest_file, 'w', encoding='utf-8') 
            #     for records_ in trainingLoss_list_epoch:
            #         json.dump(records_, output_file) 
            #         output_file.write("\n") 
            #     state = {
            #             'epoch': epoch,
            #             'state_dict': self.model.state_dict(), 
            #             'optimizer':  self.optimizer.state_dict(), 
            #             'train_loss':records["total_loss"] ,
            #             'train_psnr':sum(mae_blck)/len(mae_blck),
            #             'input_index': input_index 
            #             } 
            #     print("SAVING THE MODEL: %s/weight_%03d_%05d.pt" % (self.ckpt_path, epoch, input_index) )
            #     PATH =  '%s/weight_%03d_%05d.pt' % (self.ckpt_path, epoch, input_index)  
            #     torch.save(state, PATH) 

            pbar.set_description("Train: EP:%d [%d] Train loss:%.2f MAE:%.2f" % (epoch, input_index, trainLossCount/(input_index + 1), sum(mae_blck)/(input_index + 1)))
 
        df = pd.DataFrame(trainingLoss_list_epoch) 
        df.to_csv('%s/%s/trainloss_epoch_%03d.csv' % (self.stats_path, self.log_folder,  epoch), index=False)  

        epoch_loss = trainLossCount/num_images

        return epoch_loss
    
    def evaluation(self, validLoader, epoch):
        pbar = tqdm(validLoader)  
        self.model.eval() 

        num_images = 0 
        records = {"mae": [], "rmse": [], "pred": [], "gt": [], "valid": []}
 
        
        for data in pbar:
            idx = data[0];  X_image= data[1] ; X_Irr_token  = data[2]
 
            X_image = X_image.float().to(self.device)   
            X_Irr   = X_Irr_token.float().to(self.device)  
            
            num_images += X_image.size(0)   

            with torch.no_grad():
                X_pred      = self.model(X_image) 
 
            mae         = torch.mean(torch.abs(X_pred.view(-1,1) - X_Irr.view(-1,1))).detach()
            rmse        = torch.sqrt(torch.mean(torch.sum((X_pred.view(-1,1) - X_Irr.view(-1,1))**2))) 
            valid_loss  = self.loss_function(X_pred, X_Irr.view(-1,1)) 

            records["mae"].append(mae.item())  
            records["rmse"].append(rmse.item())  
            records["valid"].append(valid_loss.item())
            records["pred"].append(X_pred.view(-1,1))  
            records["gt"].append(X_Irr.view(-1,1))  

            pbar.set_description("EP: %d [%d]  Regression: Valid [%.2f]  MAE [%.2f] and RMSE [%.2f] " % (epoch, idx, sum(records["valid"])/len(records["valid"]), sum(records["mae"])/len(records["mae"]), sum(records["rmse"])/len(records["rmse"]))) 
        
        predictions      = torch.cat(records["pred"],  dim=0).cpu().numpy().reshape(-1) 
        target_samples   = torch.cat(records["gt"],    dim=0).cpu().numpy().reshape(-1)
        valid_loss_list  = np.array(records["valid"])


        return {"predictions"   : predictions, 
                "target_samples": target_samples,   
                "Valid_List"    : valid_loss_list   }
    #     self.rmse = np.mean(records["mae"])
    #     self.mae  = np.mean(records["rmse"]) 
    #     self.prediction = records["pred"]
    #     self.gt         = records["gt"] 

    # def compute(self):
    #     return self.rmse, self.mae