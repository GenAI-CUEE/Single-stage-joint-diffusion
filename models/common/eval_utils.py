import torch.nn.functional as F 
from torch.utils.data import DataLoader  
import torch

import numpy as np  
import math
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import itertools
from tqdm import tqdm
import pdb
import time
import csv
from datetime import datetime
import datetime
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
 

def array2datetime(array):

    array_ = array[:, ::-1]
    # minute = array[:, 0] # min
    # hour   = array[:, 1] # hour
    # day    = array[:, 2] # day
    # month  = array[:, 3] # month
    # year   = array[:, 4] # year

    datetime_object = np.array([datetime.datetime(*x) for x in array_]) 

    return datetime_object
     

def custom_print(logfile_path, message_to_print): 
    print(message_to_print)
    with open(logfile_path, 'a') as of:
        of.write(message_to_print + '\n')  


def evaluate(network,  validate_dataset, args, device, epoch, log_figures="figures", log_val="val", stats_folder_name=None):  
    
    validatingLoader  = DataLoader(validate_dataset,  batch_size=1, shuffle=False)

    total_mse, total_mae, total_psnr = 0, 0, 0
    t0 = time.time()
    predictions = []
    target_samples = []
    input_samples = []
    indices   = []
    psnr_list = []
    mse_list  = []
    mae_list  = [] 

    y_timestamp_list = []

    seq_length     = args.seq_length
    predict_length = args.predict_length

    num_samples_ = 0
    num_color_channel = network.image_channels
    img_side_len      = network.image_resize
 
    pbar = tqdm(validatingLoader)
    for i, out in enumerate(pbar, 0):

        indices.append(out[0])
        input  = out[1].float().to(device)

        if validate_dataset.name == "CUEE":
            y_timestamp = out[-1]
            y_timestamp_list.append(y_timestamp)

 
        if (args.network == "Unet") : 
            target     = out[2].float()     
            prediction = network.inference_target_sequence(input)  

        elif (args.network == "Sunset") or (args.network == "Solarnet") or (args.network == "Eclipse"): #("irr" in args.gt_list) or ("IRR" in args.gt_list): 
            X_Irr         = out[3].float().to(device)  
            target        = out[4].float()  
            prediction = network.inference_target_sequence(input, X_Irr)
        
        elif (args.network == "vqvae") or (args.network == "VQVAE"):
            target        = out[2].float().to(device)  
            prediction, _    = network.inference_target_sequence(input) 

        elif (args.network == "SkyGPT"):
            target        = out[2].float().to(device)  
            prediction     = network.inference_target_sequence(input, "recon") 

        else:
            target        = out[2].float().to(device)  
            prediction    = network.inference_target_sequence(input) 
          
        
        if ((args.network == "vqvae") or (args.network == "VQVAE"))  :
            B, P, C, H, W = prediction.shape 

            target   = target.cpu().numpy()

            prediction  = torch.clamp(prediction, -0.5,0.5) + 0.5 
            input       = torch.clamp(input, -0.5,0.5) + 0.5

            prediction = prediction.cpu().numpy() 

            rmse_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1))
            mae_batch  = mean_absolute_error(255*prediction.reshape(-1), 255*target.reshape(-1))

            psnr_batch = rmse_batch 

        elif  (args.network == "SkyGPT"):
            B, P, C, H, W = prediction.shape  
            target      = target.cpu().numpy()

            prediction  = torch.clamp(prediction, -0.5,0.5) + 0.5 
            input       = torch.clamp(input, -0.5,0.5) + 0.5

            prediction = prediction.cpu().numpy() 

            rmse_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1))
            mae_batch  = mean_absolute_error(255*prediction.reshape(-1), 255*target.reshape(-1)) 

            psnr_batch = rmse_batch

        else: 

            if (args.network == "Unet") :
                if "img-ir-log" in args.gt_list :  
                    if args.is_scale: 
                        prediction = validate_dataset.reverse_scale(prediction, which="X") 
                        target     = validate_dataset.reverse_scale(target , which="X")   
                     
                    prediction = prediction.cpu().numpy().reshape(-1) 
                    target     = target.cpu().numpy().reshape(-1) 
                        
                
                elif ("img-ir-pred" in args.gt_list):
                    if args.is_scale:
                        prediction = validate_dataset.reverse_scale(prediction , which="Y") 
                        target     = validate_dataset.reverse_scale(target , which="Y") 
                  
                    prediction = prediction.cpu().numpy().reshape(-1) 
                    target     = target.cpu().numpy().reshape(-1)  
 
                rmse_batch = root_mean_squared_error(prediction, target)  
                mae_batch = mean_absolute_error(prediction, target)  

                psnr_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1)) 

            elif  (args.network == "Sunset") or (args.network == "Solarnet") or (args.network == "Eclipse"):   
                if (prediction.ndim == 2):
                    B, C = prediction.shape 

                    if args.is_scale:  
                        prediction = validate_dataset.reverse_scale(prediction.cpu().reshape(-1) , which="Y")   
                        target   = validate_dataset.reverse_scale(target.reshape(-1) , which="Y")   

                    prediction = prediction.cpu().numpy()
                    target     = target.cpu().numpy() 

                    rmse_batch = root_mean_squared_error(prediction.reshape(-1), target.reshape(-1))  
                    mae_batch = mean_absolute_error(prediction.reshape(-1), target.reshape(-1))  

                    psnr_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1)) 

    
                elif prediction.ndim == 4:
                    B, C, H, W = prediction.shape 

                    prediction = prediction.cpu().numpy()
                    target   = target.cpu().numpy()

                    # mse_batch = np.sum((255*prediction.reshape(-1) - 255*target.reshape(-1))**2)/(B*C*H*W)   
                    # mae_batch = np.sum(np.abs(255*prediction.reshape(-1) - 255*target.reshape(-1)) )/(B*C*H*W)  

                    rmse_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1))  
                    mae_batch  = mean_absolute_error(255*prediction.reshape(-1), 255*target.reshape(-1))  
                    
                    psnr_batch = rmse_batch

                elif prediction.ndim == 5:
                    B, P, C, H, W = prediction.shape 

                    prediction = prediction.cpu().numpy()
                    target   = target.cpu().numpy()

                    # mse_batch = np.sum((255*prediction.reshape(-1) - 255*target.reshape(-1))**2)/(B*P*C*H*W)   
                    # mae_batch = np.sum(np.abs(255*prediction.reshape(-1) - 255*target.reshape(-1)) )/(B*P*C*H*W)  

                    rmse_batch = root_mean_squared_error(255*prediction.reshape(-1), 255*target.reshape(-1))  
                    mae_batch  = mean_absolute_error(255*prediction.reshape(-1), 255*target.reshape(-1))  
                    
                    psnr_batch = rmse_batch
         
        psnr = 20 * np.log10(255 / psnr_batch)  

        total_psnr += psnr
        total_mse += rmse_batch
        total_mae += mae_batch

        psnr_list.append(psnr) 
        mse_list.append(rmse_batch) 
        mae_list.append(mae_batch) 
        predictions.append(prediction)

        target_samples.append(target)
        input_samples.append(input.cpu().numpy()) 
 

        num_samples_ +=1 

        # print("VAL: EP:%d [%d] av. MSE:%.2f MAE:%.2f PSNR:%.2f" % (epoch, i, total_mse/num_samples_, total_mae/num_samples_, total_psnr/num_samples_)) 
        pbar.set_description("%s: EP:%d [%d] av. Rev. MSE:%.2f Rev. MAE:%.2f PSNR:%.2f" % (log_val, epoch, i, total_mse/num_samples_, total_mae/num_samples_, total_psnr/num_samples_))
 
    target_samples =  np.concatenate(target_samples,axis=0)  # (NxB, PredLen, 3, 64, 64) 
    input_samples  =  np.concatenate(input_samples,axis=0) # (NxB, PredLen, 3, 64, 64) 
    predictions    =  np.concatenate(predictions,axis=0) # (NxB, PredLen, 3, 64, 64) 

    if validate_dataset.name == "CUEE":  
        y_timestamp_list =  np.concatenate(y_timestamp_list,axis=0) #  

    
    if  ("img-ir-log" in args.gt_list) or ("img-ir-pred" in args.gt_list):
    
        mse_overall = np.mean((predictions.reshape(-1) - target_samples.reshape(-1))**2) 
        mae_overall = np.mean(np.abs(predictions.reshape(-1) - target_samples.reshape(-1)) ) 

        mse_overall = root_mean_squared_error(predictions.reshape(-1), target_samples.reshape(-1))  
        mae_overall = mean_absolute_error(predictions.reshape(-1), target_samples.reshape(-1))  
        
         
        Val_dict = {} 
        Val_dict["EP"]   = epoch
        Val_dict["MSE"]  = mse_overall
        Val_dict["MAE"]  = mae_overall

        print("%s @ EP %d MAE = %.2f (num samples = %d)" % (log_val, epoch, mae_overall, len(pbar))) 
       
    
    elif  (("irr" in args.gt_list) or ("IRR" in args.gt_list)) :  

        mse_overall = root_mean_squared_error(predictions.reshape(-1), target_samples.reshape(-1))  
        mae_overall = mean_absolute_error(predictions.reshape(-1), target_samples.reshape(-1))  


        Val_dict = {} 
        Val_dict["EP"]   = epoch
        Val_dict["MSE"]  = mse_overall
        Val_dict["MAE"]  = mae_overall

        print("%s @ EP %d MAE = %.2f (num samples = %d)" % (log_val, epoch, mae_overall, len(pbar))) 

    else:
        print("Validation...")      
        print('MSE per frame:{0:.2f}'.format(total_mse/num_samples_))  
        print('MAE per frame:{0:.2f}'.format(total_mae/num_samples_))
        print('PSNR per frame:{0:.2f}'.format(total_psnr/num_samples_)) 
        print('time:{0:.3f}s'.format(time.time()-t0))
        print('-'*40)

        Val_dict = {} 
        Val_dict["EP"]   = epoch
        Val_dict["MSE"]  = total_mse/num_samples_
        Val_dict["MAE"]  = total_mae/num_samples_
        Val_dict["PSNR"] = total_psnr/num_samples_

        print("%s @ EP %d PSNR = %.2f dB (num samples = %d)" % (log_val, epoch, total_psnr/num_samples_, len(pbar))) 

    
    # Val_dict["psnr_list"] = psnr_list
    # Val_dict["mse_list"] = mse_list
    # Val_dict["mae_list"] = mae_list
 
    os.makedirs("%s/%s" % (stats_folder_name, log_val), exist_ok=True) 
    csv_stats_folder_name = os.path.join(stats_folder_name, log_val, "stats_%d.csv" % epoch)

    with open(csv_stats_folder_name, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in Val_dict.items():
            writer.writerow([key, value])  

    csv_prediction_csv = os.path.join(stats_folder_name, log_val, "result_%d.csv" % epoch)


    if validate_dataset.name == "CUEE":  
        if np.ndim(y_timestamp_list) == 2:
            y_timestamp_ = array2datetime(y_timestamp_list) 
        df_pred = pd.DataFrame(zip(y_timestamp_, predictions, target_samples), columns=['datetime_array', 'prediction', 'groundtruth']) 
    else:
        df_pred = pd.DataFrame(zip(predictions, target_samples), columns=['prediction', 'groundtruth'])
    df_pred.to_csv(csv_prediction_csv)
 
    # save results 

    if  ("img-ir-log" in args.gt_list) or ("img-ir-pred" in args.gt_list)  :
        select_idx = 100 
        predictions    = predictions.reshape(-1)
        target_samples = target_samples.reshape(-1) 
        name = args.network 

        plot_timeseries(predictions, target_samples, mae_overall, name, epoch, select_idx=100,   num_samples = 258,  log_figures=log_figures, stats_folder_name=stats_folder_name)
    
    elif (("irr" in args.gt_list) or ("IRR" in args.gt_list))  :
        select_idx = 100 
        predictions    = predictions.reshape(-1)
        target_samples = target_samples.reshape(-1) 
        name = args.network 

        if validate_dataset.name == "CUEE":  
            plot_timeseries(predictions, target_samples, mae_overall, name, epoch, select_idx=100,   num_samples = 1000, datetime_x=y_timestamp_, log_figures=log_figures, stats_folder_name=stats_folder_name)
        else:
            plot_timeseries(predictions, target_samples, mae_overall, name, epoch, select_idx=100,   num_samples = 1000, log_figures=log_figures, stats_folder_name=stats_folder_name)

    else:

        select_idx = [1000] 

        if (args.network == "Skynet") :
            B, PC, H, W = predictions.shape 
            predictions    = predictions.reshape(B, predict_length, 3, img_side_len, img_side_len)  
            target_samples = target_samples.reshape(B, predict_length, 3, img_side_len, img_side_len)   
            input_samples = input_samples.reshape(B, seq_length, 3, img_side_len, img_side_len)  

            plot_gen_images(predictions, input_samples, target_samples, select_idx,  epoch, log_figures=log_figures, stats_folder_name=stats_folder_name)
        
        elif   (args.network == "vqvae") or (args.network == "SkyGPT") :
            predictions    = predictions.transpose(0, 2, 1, 3, 4)  
            target_samples = target_samples.transpose(0, 2, 1, 3, 4)   
            input_samples = input_samples.transpose(0, 2, 1, 3, 4)  

            plot_gen_images(predictions, input_samples, input_samples, select_idx,  epoch, log_figures=log_figures, stats_folder_name=stats_folder_name)
        
        else:
        
            plot_gen_images(predictions, input_samples, target_samples, select_idx,  epoch, log_figures=log_figures, stats_folder_name=stats_folder_name)
    
    return total_mse/num_samples_,  total_mae/num_samples_, predictions,  indices, target_samples, input_samples
 
 
def plot_learning(generator_train_losses, discriminator_train_losses=[], epoch=0, log_figures="figures", stats_folder_name=None):
    plt.close("all")
    
    f,ax=plt.subplots()
    ax.plot(range(len(generator_train_losses)),generator_train_losses,label="gen_loss")
    
    if len(discriminator_train_losses) > 0:
    
        ax.plot(range(len(discriminator_train_losses)),discriminator_train_losses,label='disc_loss')
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.grid()
    plt.tight_layout()

    if stats_folder_name is not None:
        os.makedirs("%s/%s" % (stats_folder_name, log_figures), exist_ok=True)
        f.savefig("%s/%s/plot_learning_%d.png" % (stats_folder_name, log_figures, epoch)) 
    else:
        f.savefig("results/plot_learning_%d.png" % epoch) 
        

def psnr_cal(prediction, target):
    if prediction.ndim == 5:
        B, P, C, H, W = prediction.shape 
        mse_batch = np.sum((255*prediction.reshape(-1) - 255*target.reshape(-1))**2)/(B*P*C*H*W)    
    elif prediction.ndim == 4:
        B, C, H, W = prediction.shape 
        mse_batch = np.sum((255*prediction.reshape(-1) - 255*target.reshape(-1))**2)/(B*C*H*W)          
    elif prediction.ndim == 3:
        C, H, W = prediction.shape 
        mse_batch = np.sum((255*prediction.reshape(-1) - 255*target.reshape(-1))**2)/(C*H*W) 
    
    psnr = 20 * np.log10(255 / np.sqrt(mse_batch))  
    return psnr
 

def PSNR_torch(prediction, target):  
    
    if target.dim() == 3:
        B, H, W = target.shape
        mse = torch.sum((target.reshape(-1) - prediction.reshape(-1)) ** 2)/(B*H*W)   

    elif target.dim() == 4:
        B, C, H, W = target.shape
        mse = torch.sum((target.reshape(-1) - prediction.reshape(-1)) ** 2)/(B*C*H*W)   
        
    elif target.dim() == 5:
        B, t, C, H, W = target.shape    
        mse = torch.sum((target.reshape(-1) - prediction.reshape(-1)) ** 2)/(B*t*C*H*W)   
    
    max_pixel = 255 
    psnr      = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr



def plot_timeseries(predictions, target_samples, mae_overall, name_method, epoch, select_idx, datetime_x = None, num_samples = 258, log_figures="figures", stats_folder_name=None):
     
    predict = predictions[select_idx: select_idx + num_samples] 
    target  = target_samples[select_idx: select_idx + num_samples] 

    time_x = np.arange(num_samples)  
    time_x_tick = np.arange(0, num_samples, 36)
    
    # if datetime_x is None:
    #     time_x = np.arange(num_samples)  
    #     time_x_tick = np.arange(0, num_samples, 36)
    # else:
    #     #Don't want to do this option because the time stamp was jumpy as you randomize the samples...
    #     time_x = datetime_x[select_idx: select_idx + num_samples] 
    #     time_x_tick = time_x[np.arange(0, num_samples, 36)]
 
    fig = plt.figure(figsize=(15,8))
    plt.plot(time_x,  target, label = "Actual",  color="black")
    plt.plot(time_x, predict, label = "%s" % name_method, color="red")
    plt.xticks(time_x_tick, rotation=45, ha='right')
    plt.title("EP [%d] MAE %0.2f @ Horrizon %d mins ahead" % (epoch, mae_overall, 15))
    plt.ylabel("$Watt/m^2$")
    plt.legend()  
    plt.grid(True)
    plt.tight_layout()   

    if stats_folder_name is not None:
        os.makedirs("%s/%s" % (stats_folder_name, log_figures), exist_ok=True)
        plt.savefig("%s/%s/plot_timeseries_%d.png" % (stats_folder_name, log_figures, epoch)) 
    else:
        plt.savefig("results/plot_timeseries_%d.png" % epoch) 
 

def plot_gen_images(predictions, images_input, images_target, select_idx, epoch,  log_figures="figures", stats_folder_name=None):
 
    predictions = predictions.transpose((0,1,3,4,2))
    images_input = images_input.transpose((0,1,3,4,2))
    images_target = images_target.transpose((0,1,3,4,2))

    for i in range(len(select_idx)):
        #print("-"*50,"sample ",str(i+1), "-"*50)
        f, ax = plt.subplots(2,8)
        f.subplots_adjust(wspace=0, hspace=0.15)
        f.set_size_inches(24,6)

        psnr_0 = psnr_cal(predictions[select_idx[i], 0, :, :, :], images_target[select_idx[i],0,:,:,:])
        psnr_2 = psnr_cal(predictions[select_idx[i], 2, :, :, :], images_target[select_idx[i],2,:,:,:])
        psnr_4 = psnr_cal(predictions[select_idx[i], 4, :, :, :], images_target[select_idx[i],4,:,:,:]) 
        psnr_7 = psnr_cal(predictions[select_idx[i], 7, :, :, :], images_target[select_idx[i],7,:,:,:]) 
        
 
        ax[0,0].imshow(images_input[select_idx[i],0,:,:,:])
        ax[0,0].set_title("$X[t+=0]$")
        ax[0,1].imshow(images_input[select_idx[i],2,:,:,:])
        ax[0,1].set_title("$X[t+=2]$")
        ax[0,2].imshow(images_input[select_idx[i],4,:,:,:])
        ax[0,2].set_title("$X[t+=4]$")
        ax[0,3].imshow(images_input[select_idx[i],7,:,:,:])
        ax[0,3].set_title("$X[t+=7]$")

        ax[0,4].imshow(images_target[select_idx[i],0,:,:,:])
        ax[0,4].set_title("$Y_{gt}[t+=0]$")
        ax[0,5].imshow(images_target[select_idx[i],2,:,:,:])
        ax[0,5].set_title("$Y_{gt}[t+=2]$")
        ax[0,6].imshow(images_target[select_idx[i],4,:,:,:])
        ax[0,6].set_title("$Y_{gt}[t+=4]$")
        ax[0,7].imshow(images_target[select_idx[i],7,:,:,:])
        ax[0,7].set_title("$Y_{gt}[t+=7]$")         

        ax[1,4].imshow(predictions[select_idx[i], 0, :, :, :])
        ax[1,4].set_title("$Y_{pred}[t+=0] [%0.2f dB]$" % psnr_0)
        ax[1,5].imshow(predictions[select_idx[i], 2, :, :, :])
        ax[1,5].set_title("$Y_{pred}[t+=2] [%0.2f dB]$" % psnr_2)
        ax[1,6].imshow(predictions[select_idx[i], 4, :, :, :])
        ax[1,6].set_title("$Y_{pred}[t+=4] [%0.2f dB]$" % psnr_4)
        ax[1,7].imshow(predictions[select_idx[i], 7, :, :, :])
        ax[1,7].set_title("$Y_{pred}[t+=7] [%0.2f dB]$" % psnr_7)
   
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[0,2].axis('off')
        ax[0,3].axis('off')
        ax[0,4].axis('off')
        ax[0,5].axis('off')
        ax[0,6].axis('off')
        ax[0,7].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        ax[1,2].axis('off')
        ax[1,3].axis('off')
        ax[1,4].axis('off')
        ax[1,5].axis('off')
        ax[1,6].axis('off')
        ax[1,7].axis('off')
 
        plt.tight_layout()  

        if stats_folder_name is not None:
            os.makedirs("%s/%s" % (stats_folder_name, log_figures), exist_ok=True)
            f.savefig("%s/%s/image_%d.png" % (stats_folder_name, log_figures, epoch)) 
        else:
            f.savefig("results/image_%d.png" % epoch) 
