# %%
import glob
import pdb
import os

import csv   
import torch
import numpy as np
import pandas as pd
import h5py

import ast
import pytz

import imageio.v3 as iio
from PIL import Image  
from matplotlib import pyplot as plt

from math import *
import calendar

from datetime import date,  time
import datetime

from tqdm import tqdm 
from skimage.restoration import denoise_nl_means, estimate_sigma  
import cv2
import yaml
from PIL import Image

def myresize(image, newsize):
    image = Image.fromarray(image)
    image = image.resize((newsize, newsize))
    return np.array(image)
 
def save_dict_as_h5file(dict, filename):
    hf = h5py.File(filename, "w")   
    for key_, value_ in dict.items(): 
        hf.create_dataset(key_, data=value_ ) 

    hf.close()

def PSNR_calculation(original, reconstructed, dim=4):
    """ 
    To calculate MAE, RMSE, and PSNR of a reconstructed image or reconstructed images in a batch
    
    Input:
        original: np.array uint8 image  (0-255)
        reconstructed: np.array uint8 image  (0-255)
        dim: dimension of the two images. 
    
    Output:
        mae : the mean absolute value of the error between the reconstructed and the originals images.
        rmse  : the root mean square value of the error between the reconstructed and the originals images.
        psnr : peak signal to noise ratio of the reconstructed images compared to the originals.
    
    """

    if dim == 4:
        mae = np.sum(np.abs(original - reconstructed))/(original.shape[0]*original.shape[1]*original.shape[2]*original.shape[3]) 
        mse = np.sum((original - reconstructed) ** 2)/(original.shape[0]*original.shape[1]*original.shape[2]*original.shape[3])
    else:
        mae = np.sum(np.abs(original - reconstructed))/(original.shape[0]*original.shape[1]*original.shape[2]) 
        mse = np.sum((original - reconstructed) ** 2)/(original.shape[0]*original.shape[1]*original.shape[2])

    if(mse == 0):  
        # MSE is zero means there is no noise in the images.
        # PSNR will be 20 * log10(max_pixel / sqrt(0)) => inf. 
        # So, we replace this value with 100 which is high enough. 
        psnr = 100
        rmse = 0 
    else:
        max_pixel = 255
        psnr = 20 * log10(max_pixel / sqrt(mse))
        rmse = np.sqrt(mse) 

    return mae, rmse, psnr

def cv2_medfilter(img_input:np.ndarray, ksize:int):
    img_BGR      = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    img_filtered = cv2.medianBlur(img_BGR, ksize)
    img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
    return img_filtered
 

class SaveImage:
    def __init__(self, checkpoint_path:str, id_name):
        self.tracker_path = os.path.join(checkpoint_path, id_name) 
        os.makedirs(self.tracker_path, exist_ok = True) 

    def save_per_image(self, image, image_path:str): 

        if isinstance(image, torch.Tensor): 
            """
            If an image is a torch tensor, we assume that the size is 3xHxW or HxWx3,
                It can only be a `torch.float` with value -1 to 1, or `torch.uint8` with value 0 to 255.
            
            """
            shape = image.shape

            if len(shape) == 3 and shape[0] == 3:
                image = image.permute(1, 2, 0)

            elif len(shape) == 3 and shape[2] == 3:
                image

            else:
                print("If an image is a torch tensor, we assume that the size is 3xHxW or HxWx3")
                print("It can only be a `torch.float` with value -1 to 1, or `torch.uint8` with value 0 to 255") 
                KeyError


            if (image.dtype == torch.float) and (image.max() <= 1) and (image.min() >= -1):
                ndarr = ((image.float()/2+0.5)*255).to("cpu", torch.uint8).numpy()

            elif (image.dtype == torch.uint8) and (image.max() <= 256) and (image.min() >= 0): 
                ndarr = image.to("cpu").numpy()
            
            else:
                print("If an image is a torch tensor, we assume that the size is 3xHxW or HxWx3")
                print("It can only be a `torch.float` with value -1 to 1, or `torch.uint8` with value 0 to 255") 
                KeyError 
            
        
        elif isinstance(image, np.ndarray): 
            """
            If an image is np.ndarray, we assume that the size is 3xHxW or HxWx3,
                It can only be a `np.floating` with value -1 to 1, or `np.uint8` with value 0 to 255.
            
            """
            shape = image.shape

            if len(shape) == 3 and shape[0] == 3:
                image = np.transpose(image, (1, 2, 0)) 

            elif len(shape) == 3 and shape[2] == 3:
                image

            else:
                print("If an image is a np.ndarray, we assume that the size is 3xHxW or HxWx3")
                print("It can only be a `np.floating` with value -1 to 1, or `np.uint8` with value 0 to 255") 
                KeyError

            if (image.dtype == np.floating) and (image.max() <= 1) and (image.min() >= -1):
                ndarr = ((image/2+0.5)*255) 

            elif (image.dtype == np.uint8) and (image.max() <= 256) and (image.min() >= 0): 
                ndarr = image 

            else:
                print("If an image is a np.ndarray, we assume that the size is 3xHxW or HxWx3")
                print("It can only be a `np.floating` with value -1 to 1, or `np.uint8` with value 0 to 255") 
                KeyError      

        im    = Image.fromarray(ndarr)
        im.save(image_path)

    def save_image_list(self, image_list:list[torch.Tensor], datetime_list:list, key="GENIMG"): 

        for index, (img_, dt_) in enumerate(zip(image_list, datetime_list)):

            image_path = os.path.join(self.tracker_path, "%s-%d-%s.png" % (key, index, dt_.strftime('%Y-%m-%d_%H-%M-%S')))
            self.save_per_image(img_, image_path)



def process_images(gt_image_array, mask, radius,  new_size=None):
    shape               = gt_image_array.shape 
    img_pil_gt          = Image.fromarray(gt_image_array)
    img_width, img_height = img_pil_gt.size
    min_dis             = min(img_width, img_height) 

    center_x            = img_width // 2
    center_y            = img_height // 2 

    left                = center_x - radius
    right               = center_x + radius

    top                 = center_y - radius
    bottom              = center_y + radius
    
    masked_image         = np.copy(gt_image_array)
    masked_image[~mask]  = 0 
    masked_image         = Image.fromarray(masked_image)  
    masked_image_cropped = masked_image.crop((left, top, right, bottom))


    if new_size is not None:
        maskedimg_width, maskedimg_height = masked_image_cropped.size  
        shrinkFactor          = new_size /min([maskedimg_width,maskedimg_height])   

        masked_image_cropped = masked_image_cropped.resize((round(maskedimg_width*shrinkFactor),round(maskedimg_height*shrinkFactor)))
        maskedimg_width, maskedimg_height = masked_image_cropped.size  
        masked_image_cropped = np.array(masked_image_cropped)  
    
    else:
        maskedimg_width, maskedimg_height = masked_image_cropped.size  
        masked_image_cropped = np.array(masked_image_cropped) 

    return masked_image_cropped, masked_image




def get_data_from_quartile(sirta_cloud_and_data, sirta_images, main_folder, sirta_cloud_and_mask = None,  cloud_attribute= 'cloud_cover', operation="Lower", quartile=0.25):   

    
    if cloud_attribute == "cloud_cover":
        if operation == "Lower":
            q_path = os.path.join(main_folder, "cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] <= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]
        
        elif  operation == "Higher":
            q_path = os.path.join(main_folder, "cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] >= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]

    elif cloud_attribute == "corrected_cloud_cover":
        if operation == "Lower":
            q_path = os.path.join(main_folder, "corrected_cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] <= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]
        
        elif  operation == "Higher":
            q_path = os.path.join(main_folder, "corrected_cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] >= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]

    
    elif cloud_attribute == "Elifan+Detect":
        if operation == "Lower":
            q_path = os.path.join(main_folder, "Elifan+Detect_cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] <= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]
        
        elif  operation == "Higher":
            q_path = os.path.join(main_folder, "Elifan+Detect_cloud_cover_%s_Q%02d" % (operation, quartile*100))
            os.makedirs(q_path, exist_ok=True)

            q_data = sirta_cloud_and_data[sirta_cloud_and_data[cloud_attribute] >= sirta_cloud_and_data[cloud_attribute].quantile(quartile)]


    q_data["Datetime"] = pd.to_datetime(q_data["Datetime"]) 
    q_dates = q_data["Datetime"].tolist()

    datetime_list            = sirta_images["Datetime"][sirta_images["Datetime"].isin(q_dates)].tolist()
    sirta_images_h5img_list  = sirta_images["h5img"][sirta_images["Datetime"].isin(q_dates)].tolist()
    
    cloud_cover_list = q_data[cloud_attribute].tolist()
    irradiance_list  = q_data["I"].tolist()
    iclr_list        = q_data["Iclr"].tolist()

    if sirta_cloud_and_mask is not None:
         
        assert sirta_cloud_and_mask["Datetime"].isin(datetime_list).any()             

        for date_, image_, cloud_cover_, irrad_, iclr_ in zip(datetime_list, sirta_images_h5img_list, cloud_cover_list, irradiance_list, iclr_list): 
            
            cloud_mask_ = sirta_cloud_and_mask["cloud_mask"][sirta_cloud_and_mask["Datetime"] == date_].values[0] 
 
            img        = Image.fromarray(image_, 'RGB')
            cloud_mask = Image.fromarray(cloud_mask_) 

            date_name = date_.strftime('%X-%Y-%m-%d')
            if cloud_attribute == "cloud_cover":
                img.save(os.path.join(q_path, '%s-Cl%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_))) 
            elif cloud_attribute == "corrected_cloud_cover":
                img.save(os.path.join(q_path, '%s-CorrCl%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_))) 
            elif cloud_attribute == "Elifan+Detect":
                img.save(os.path.join(q_path, '%s-Elifan+Detect%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_)))  

            cloud_mask.save(os.path.join(q_path, '%s-Mask%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_)))  
    
    else:
        for date_, image_, cloud_cover_, irrad_, iclr_ in zip(datetime_list, sirta_images_h5img_list, cloud_cover_list, irradiance_list, iclr_list): 
            img = Image.fromarray(image_, 'RGB')
            date_name = date_.strftime('%X-%Y-%m-%d')
            if cloud_attribute == "cloud_cover":
                img.save(os.path.join(q_path, '%s-Cl%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_))) 
            elif cloud_attribute == "corrected_cloud_cover":
                img.save(os.path.join(q_path, '%s-CorrCl%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_))) 
            elif cloud_attribute == "Elifan+Detect":
                img.save(os.path.join(q_path, '%s-Elifan+Detect%.2f-Irr%.2f-Iclr%.2f.png' % (date_name, cloud_cover_, irrad_, iclr_))) 

    return q_dates  


def scatter_plot(sirta_cloud_and_data, x="Iclr", y="I", ylim_=None, xlim_=None, alpha=0.5, size=20):  
    plt.scatter(sirta_cloud_and_data[x], sirta_cloud_and_data[y], alpha=alpha, s=size)
    plt.xlabel(x)
    plt.ylabel(y)
    if ylim_ is not None:
        plt.ylim(ylim_)
    if xlim_ is not None:
        plt.xlim(xlim_)
    plt.show() 


def my_eval(text):
    if type(text) == str:
        try:
            extracted_list = ast.literal_eval(text) 
        except:
            extracted_text = text.split("[")[1].split("]")[0]
            text_elements =  extracted_text.split(",")
            extracted_list = []
            for text_ in text_elements:
                try:
                    extracted_list.append(eval(text_))
                except:
                    pdb.set_trace()
    else:
        extracted_list = text
    return extracted_list



def crop_center(img,cropx,cropy):
    _, y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    return img[:, starty:starty+cropy,startx:startx+cropx, :]

def file_name2datetime(file_, with_time = False):
    basename = os.path.basename(file_)  
    year_  = int(basename.split("-")[2])
    month_ = int(basename.split("-")[0])
    day_   = int(basename.split("-")[1])  
    
    if with_time:
        hour_  = int(basename.split("-")[3])
        min_   = int(basename.split("-")[4])
        sec_   = int(basename.split("-")[5].split(".h5")[0]) 
        datetime_ = datetime.datetime(year_, month_, day_, hour_, min_, sec_)
    else:
        datetime_ = datetime.datetime(year_, month_, day_)
 
    return datetime_

def filename2date(list_of_files):
    days_list_per_month_all = []
    for file_ in list_of_files:
        datetime_ = file_name2datetime(file_) 
        days_list_per_month_all.append(datetime_) 
    
    return days_list_per_month_all




def extract_dict_dayslist(file_list):
    file2datetime = {}
 
    days_list_per_month_all = [] 
 
    for file_ in file_list:
        datetime_ = file_name2datetime(file_, with_time=False) 
        days_list_per_month_all.append(datetime_)  

  
    days_list_per_month_set = set(days_list_per_month_all)
    days_list_per_month     = list(days_list_per_month_set)  
    date2files = {}
    for date in days_list_per_month:

        date2files[date] = [] 
        for file_ in file_list: 
            file_datetime = file_name2datetime(file_, with_time=False)   
            if date == file_datetime:
                date2files[date].append(file_)  

    for file_ in file_list:
        file2datetime[file_] = file_name2datetime(file_, with_time=True) 

    return days_list_per_month, date2files, file2datetime

def get_index_training_valid_testing(total_num, training_num, valid_num):
    total_index = np.arange(total_num)
 
    training_index    = total_index[np.random.choice(total_num, training_num, replace=False)]  
    nottraining_index = np.setdiff1d(total_index, training_index)  
 
    valid_index      = nottraining_index[np.random.choice(nottraining_index.shape[0], valid_num, replace=False)]  
    testing_index    = np.setdiff1d(np.setdiff1d(total_index, training_index), valid_index)
  
    return training_index, valid_index, testing_index

def imread2nparray(unzip_image_dir, Date_, image_name, im_resize=False, new_size = 480):

    gt_image_array    = iio.imread("%s/%s/%s/%s" % (unzip_image_dir, Date_, Date_, image_name)) 

    if im_resize: 

        shape               = gt_image_array.shape 
        nr, nc              = shape[0], shape[1]
        shrinkFactor        = new_size /max([nr,nc])  
        img_pil_gt          = Image.fromarray(gt_image_array)
        img_pil_gt          = img_pil_gt.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
        gt_image_array_pil  = np.array(img_pil_gt)

        image_arrays  = np.transpose(gt_image_array_pil,(2,1,0))  

    else:
        image_arrays  = np.transpose(gt_image_array,(2,1,0)) 

    return image_arrays


def adding_images_2_dataframe(dataframe, unzip_image_dir, h5folder_name, im_resize=False, new_size = 480):
    
    images_array = [] 

    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
    
        file_name        = row["Name"]  
        date_time_string = row["Datetime"].strftime("%m-%d-%Y-%H-%M-%S")
        Date_            = row["Date"].replace("-","_")
        Irradiance_      = row["Irradiance_30 (W/m2)"] 
  
        image_array_trans = imread2nparray(unzip_image_dir, Date_, file_name, im_resize=im_resize, new_size=new_size )

        hf = h5py.File("%s/%s.h5" % (h5folder_name, date_time_string), 'w')
        hf.create_dataset('X', data=image_array_trans)
        hf.create_dataset('Y', data=Irradiance_)
        hf.close()  
   
        pdb.set_description("[%d/%d] %s" % (index, dataframe.shape[0], file_name))
     
    return dataframe


def convert_df_to_ref(input_datetime):
    df_numer = {}
                
    df_numer['year']  = input_datetime.apply(lambda row: row.year) 
    df_numer['month'] = input_datetime.apply(lambda row: row.month) 
    df_numer['day']   = input_datetime.apply(lambda row: row.day)  
    df_numer['hour']  = input_datetime.apply(lambda row: row.hour)  
    df_numer['min']   = input_datetime.apply(lambda row: row.minute) 

    df_numer_array = np.stack([df_numer['hour'].values, df_numer['min'].values, df_numer['day'].values, df_numer['month'].values, df_numer['year'].values], axis=1) 
 
    return df_numer_array


def read_h5toDf(filepath):
    file = h5py.File(filepath)
    out_dict = {}
    for key,val in file.items():
        if type(val) == h5py._hl.dataset.Dataset:
            out_dict[key] = np.array(val) 

    df_images          = pd.DataFrame() 

    for key_ in out_dict.keys():
        if (key_ == 'Datetime'):
            datetime_arry      = out_dict['Datetime'] 
            datetime_arry2list = list(datetime_arry) 
            datetime_list      = [pd.Timestamp(datetime.datetime(arr_[4], arr_[3], arr_[2], arr_[0], arr_[1])) for arr_ in datetime_arry2list] 
            df_images["Datetime"] = datetime_list 
        else:
            genimage_arry = out_dict[key_]  
            genimage_arry2list = list(genimage_arry)  
            df_images[key_] = genimage_arry2list 
 
    return df_images

def numer2dataframe(input_datetime):
    df_numer = {}
                
    df_numer['year']  = input_datetime.apply(lambda row: row.year) 
    df_numer['month'] = input_datetime.apply(lambda row: row.month) 
    df_numer['day']   = input_datetime.apply(lambda row: row.day)  
    df_numer['hour']  = input_datetime.apply(lambda row: row.hour)  
    df_numer['min']   = input_datetime.apply(lambda row: row.minute) 

    df_numer_array = np.array([df_numer['min'].values[0], df_numer['hour'].values[0], df_numer['day'].values[0], df_numer['month'].values[0], df_numer['year'].values[0]]) 
 
    return df_numer_array

def numer2dataframe_toindex(input_datetime):
    df_numer = {}
                
    df_numer['year']  = pd.DatetimeIndex(input_datetime).year 
    df_numer['month'] = pd.DatetimeIndex(input_datetime).month 
    df_numer['day']   = pd.DatetimeIndex(input_datetime).day 
    df_numer['hour']  = pd.DatetimeIndex(input_datetime).hour 
    df_numer['min']   = pd.DatetimeIndex(input_datetime).minute 

    df_numer_array = np.array([df_numer['min'].values, df_numer['hour'].values, df_numer['day'].values, df_numer['month'].values, df_numer['year'].values]) 
    return df_numer_array

def imagelist2statdata(imagename_list, start_date = (2023, 3, 15), stop_date = (2023, 11, 3), localize_datetime=None, image_timezone="Local"):
    # Putting each image into dataframe:
    # | "Name"     |      "Datetime"         |  "Date" |  "Date_str": '%04d-%02d-%02d'|  "Minutes" |  "Time" | "Year" | "Month" | "day" | "hour" | "min" |
    #  ----------------------------------------------------------------------------------------------------------------------------------------------
    # | image_name |  datetime.fromisoformat |  .... 
    # 
    imagename_list.sort()
    START_DATE =  date.fromisoformat('%04d-%02d-%02d' % start_date)
    STOP_DATE  =  date.fromisoformat('%04d-%02d-%02d' % stop_date) 

    data_frame = []
    pbar = tqdm(imagename_list)
    for image_path_name in pbar:

        image_name = os.path.basename(image_path_name)
        date_time  = image_name.split("_01.jpg")[0] 
        
        parent_dir = os.path.dirname(image_path_name)
        image_name_underexpos = "%s_03.jpg" % date_time
        if not(os.path.exists(os.path.join(parent_dir, image_name_underexpos))):
            print("File not found: %s" % os.path.join(parent_dir, image_name_underexpos))
            continue 

        year       = int(date_time[:4]) 
        month      = int(date_time[4:6])
        day        = int(date_time[6:8]) 
        hour       = int(date_time[8:10]) 
        minute     = int(date_time[10:12]) 

        image_name_underexpos = "%s_03.jpg" % date_time

        if image_timezone == "Local":

            DateTime_ISO = '%04d-%02d-%02dT%02d:%02d:00' % (year, month, day, hour, minute) 

        elif image_timezone == "UTC":

            DateTime_ISO = '%04d-%02d-%02dT%02d:%02d:00Z' % (year, month, day, hour, minute) 
        
        DateTime = datetime.datetime.fromisoformat(DateTime_ISO)  

        if localize_datetime is not None: 
            localtz  = pytz.timezone(localize_datetime) 
            DateTime = DateTime.astimezone(tz=localtz).replace(tzinfo=None)    
            year   = DateTime.year
            month  = DateTime.month
            day    = DateTime.day
            hour   = DateTime.hour
            minute = DateTime.minute
 

        Date     = date.fromisoformat('%04d-%02d-%02d' % (year, month, day))
        Time     = time.fromisoformat('%02d:%02d:00'   % (hour, minute)) 
        
        if Date >= START_DATE  and  Date <= STOP_DATE :

            time_minute = Time.hour * 60 + Time.minute 
            data_frame.append({"Name":  image_name, "Name_underexpos":  image_name_underexpos,  "Datetime": DateTime, "Date": Date, "Date_str": '%04d-%02d-%02d' % (year, month, day), "Minutes": time_minute, "Time":Time, "Year": year, "Month":month, "day":day, "hour":hour, "min":minute}) 
            pbar.set_description("image_name: %s - Datetime: %s" % (image_name, DateTime))

    data_frame = pd.DataFrame(data_frame) 
    return data_frame


def statdata2selecthours(data_frame, start_hour = (8, 00),  end_hour = (16, 00)):
    # Putting each image into dataframe:
    # | "Name"     |      "Datetime"         |  "Date" |  "Date_str": '%04d-%02d-%02d'|  "Minutes" |  "Time" | "Year" | "Month" | "day" | "hour" | "min" |
    #  ----------------------------------------------------------------------------------------------------------------------------------------------
    # | image_name |  datetime.fromisoformat |  .... 
    # |
    # |               selecting between 8:00 - 16:00
    # |    
    start_hour =  time.fromisoformat('%02d:%02d:00'   % start_hour) 
    end_hour   =  time.fromisoformat('%02d:%02d:00'   % end_hour) 

    selection       = (data_frame["Time"] >= start_hour).values * (data_frame["Time"] <= end_hour).values 
    data_frame_noon = data_frame.iloc[selection].copy() 

    return data_frame_noon

def condition2numerical(dataframe):
    mapping2number = {"clr": 0, "partly_cloudy": 1, "cloudy": 2} 
    list_of_values = dataframe.values.tolist()
    numerical_list = [mapping2number[label] for label in list_of_values] 
    return np.array(numerical_list)


def merge_with_irradiance_csv(data_frame_noon, csv_2023, is_h5seq=True): 

    if is_h5seq:
        Fields_CSV    = ["Datetime", "I",  "H", "P",  "cos_zna", "Iclr", "k_bar", "condition"] 
    else: 
        Fields_CSV    = ["Datetime", "I", "Diff", "C", "H", "P",  "cos_zna", "Iclr", "Idhi",  "k_bar", "condition"] 
    
    cuee_csv      = csv_2023.loc[:, Fields_CSV]
    cuee_csv      = cuee_csv[~cuee_csv.isnull().any(axis=1)] 
    cuee_csv      = cuee_csv.rename(columns={'Datetime': 'Datetimestr'}) 
     

    cuee_csv_list = []
    for index, row in cuee_csv.iterrows():
        cuee_csv_list.append(datetime.datetime.fromisoformat(row["Datetimestr"])) 

    cuee_csv["Datetime"] = cuee_csv_list
    merged_df = pd.merge(data_frame_noon, cuee_csv, on='Datetime', how='inner') # Now we have the merged dataframe 

    merged_df["numerical_condition"] = condition2numerical(merged_df["condition"])
 
    #merged_df.to_csv('merged_imagenames_and_irradiance.csv')

    res = [m_ for m_ in merged_df["Month"].values.tolist() if not(str(m_).isnumeric())]

    if len(res) == 0:
        print("all numerical values") 

    return merged_df


def divide2frames(merged_df, TimeRes, input_samples, predict_length  ):
 
    print(input_samples)

    unique_days_list = list(np.unique(merged_df["Date_str"]))  

    FileName_list = []
    Month_list = [] 
    Year_list = [] 

    pbar = tqdm(unique_days_list)
 
    for unique_day_id, unique_day in enumerate(pbar):

 
        data_per_day   = merged_df.iloc[merged_df["Date_str"].values == unique_day]
        data_per_Month = merged_df.iloc[merged_df["Date_str"].values == unique_day].Month.values[0] 
        data_per_Year  = merged_df.iloc[merged_df["Date_str"].values == unique_day].Year.values[0]

        print("data_per_Month = %d" % data_per_Month)
         
        Month_list.append(data_per_Month)

        Year_list.append(data_per_Year)
 
        # data_per_day contain the data for that day ... 
        pbar2 = tqdm(list(data_per_day["Minutes"].values)) 

        for start_time_id, start_time in enumerate(pbar2): 

            # ************ start_time is used in tracking the time-marking ***************  
            skip_incomplete_sequence = False

            for time_step in range(input_samples+predict_length+1):  
 
                if (start_time + time_step*TimeRes not in (data_per_day["Minutes"].values)):
                    skip_incomplete_sequence = True 
                          
            if skip_incomplete_sequence:
                continue

            DateTrain_list = []
            NameTrain_list = [] 
            IrradTrain_list = []
            TimeArrTrain_list = []

            cos_zna_train_list = []
            #AM_train_list   = []
            Iclr_train_list = []
            k_bar_train_list = []
            condition_train_list = []

            #print("Input features: ") 
            for time_step in range(0, input_samples): 

                time_  =  start_time + time_step*TimeRes     # what about between hours 
                 
                DateInput_ = data_per_day.iloc[data_per_day["Minutes"].values == time_].Datetime.values[0]    # Used as file names  
                NameInput_ = data_per_day.iloc[data_per_day["Minutes"].values == time_].Name.values[0] 
                 
                IrradInput_ = data_per_day.iloc[data_per_day["Minutes"].values == time_].I.values.item()
                
                TimeArrInput_ = numer2dataframe(data_per_day.iloc[data_per_day["Minutes"].values == time_].Datetime) 
    
                DateTrain_list.append(DateInput_)
                NameTrain_list.append(NameInput_)
                IrradTrain_list.append(IrradInput_)
                 
                TimeArrTrain_list.append(TimeArrInput_)
 
                cos_zna_train_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].cos_zna.values.item())
                #AM_train_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].AM.values.item())
                Iclr_train_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].Iclr.values.item())
                k_bar_train_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].k_bar.values.item())
                condition_train_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].numerical_condition.values.item())
 
                #print(f"Loop 1 time_step:{time_step} time_:{time_} DateInput_:{DateInput_} TimeArrInput_:{TimeArrInput_} NameInput_:{NameInput_} | IrradInput_:{IrradInput_}") 
                #print("[%d-%d] %s [%.02f]" %(time_step, time_, NamePred_, IrradPred_ )) 

            DatePred_list = []
            NamePred_list = []
            IrradPred_list = []
            TimeArrPred_list = []

            cos_zna_pred_list = []
            #AM_pred_list   = []
            Iclr_pred_list = []
            k_bar_pred_list = []
            condition_pred_list = []

            #print("Prediction: ")
            for time_step in range(input_samples, input_samples + predict_length): 

                time_  =  start_time + time_step*TimeRes   
 
                DatePred_    = data_per_day.iloc[data_per_day["Minutes"].values == time_].Datetime.values[0]    # Used as file names  
                NamePred_    = data_per_day.iloc[data_per_day["Minutes"].values == time_].Name.values[0]  
                
                IrradPred_   = data_per_day.iloc[data_per_day["Minutes"].values == time_].I.values.item()
                TimeArrPred_ = numer2dataframe(data_per_day.iloc[data_per_day["Minutes"].values == time_].Datetime)

                DatePred_list.append(DatePred_)
                NamePred_list.append(NamePred_)
                
                IrradPred_list.append(IrradPred_)
                TimeArrPred_list.append(TimeArrPred_) 
 
                cos_zna_pred_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].cos_zna.values.item())
                #AM_pred_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].AM.values.item())
                Iclr_pred_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].Iclr.values.item())
                k_bar_pred_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].k_bar.values.item())
                condition_pred_list.append(data_per_day.iloc[data_per_day["Minutes"].values == time_].numerical_condition.values.item())
                
                #print(f"Loop 2 time_step:{time_step} time_:{time_} DateInput_:{DatePred_} TimeArrPred_:{TimeArrPred_} NamePred_:{NamePred_} | IrradPred_:{IrradPred_}") 
                #print("[%d-%d] %s [%.02f]" %(time_step, time_, NamePred_, IrradPred_ ))  
   
             
            FileName_list.append({  "Image_Files":     NameTrain_list, 
                                    "X_Irrad":         IrradTrain_list,  
                                    "X_DateTime":      DateTrain_list,
                                    "X_DateTimeArray": TimeArrTrain_list,

                                    "X_cos_zna":        cos_zna_train_list,
                                    #"X_AM":             AM_train_list,
                                    "X_Iclr":           Iclr_train_list,
                                    "X_k_bar":          k_bar_train_list,
                                    "X_condition":      condition_train_list,
                                    

                                    "GT_Files":         NamePred_list, 
                                    "GT_Irrad":         IrradPred_list,  
                                    "GT_DateTime":      DatePred_list,  
                                    "GT_DateTimeArray": TimeArrPred_list, 

                                    "GT_cos_zna":        cos_zna_pred_list,
                                    #"GT_AM":             AM_pred_list,
                                    "GT_Iclr":           Iclr_pred_list,
                                    "GT_k_bar":          k_bar_pred_list,
                                    "GT_condition":      condition_pred_list,

                                    "Date":              unique_day, 
                                    "DateTime_IP":       pd.to_datetime(DateTrain_list[-1]).strftime("%m-%d-%Y-%H-%M-%S"), # Used as file names  
                                    'Month':             data_per_Month})  
                
 
                
            all_inputfile_in_list = ','.join(map(str, NameTrain_list[:2]))
            all_predfile_in_list  = ','.join(map(str, NamePred_list[:2])) 
             
            pbar2.set_description("%d-%s : Tr[%d]: %s / Pred[%d]: %s"  %  (unique_day_id, unique_day,  len(NameTrain_list), all_inputfile_in_list, len(NamePred_list), all_predfile_in_list))
         
        pbar.set_description("%d-%s: #Total H5Files= %d"  %  (unique_day_id, unique_day, len(FileName_list)))

    FileName_list = pd.DataFrame(FileName_list)
  
    return  FileName_list, Month_list, Year_list


def image2h5files(FileName_list, Month_list, Year_list, h5_dir, image_dir, im_resize, new_size ):
     
    
    Month_list_unique = list(set(Month_list))
    Year_list_unique  = list(set(Year_list)) # usually we have only one year

    year_ = Year_list_unique[-1]

    list_all_files = []
    dict_list_all_files_per_month = {month_ : [] for month_ in Month_list_unique} 

    pbar_new = tqdm(FileName_list.iterrows())
    

    for index, row in pbar_new:
 
        Date_               = row["Date"].replace("-","")  

        input_image_names   = my_eval(row["Image_Files"] ) 
        gt_image_names      = my_eval(row["GT_Files"])
 
        X_DateTime          = my_eval(row["X_DateTime"]) 
        X_DateTime_Array    = np.transpose(np.stack(numer2dataframe_toindex(X_DateTime), axis=0))       #np.stack(my_eval(row["X_DateTimeArray"]),axis=0) 
 
        Y_DateTime          = my_eval(row["GT_DateTime"])  
        GT_DateTimeArray    = np.transpose(np.stack(numer2dataframe_toindex(Y_DateTime), axis=0))   
 
        date_time_string    = row["DateTime_IP"]
        month_              = row['Month']
  
        X_Irrad             = np.array(my_eval(row["X_Irrad"]))  
        GT_Irrad            = np.array(my_eval(row["GT_Irrad"]))  # Y_Irrad
        
        X_cos_zna           = np.array(my_eval(row["X_cos_zna"]))  
        #X_am                = np.array(my_eval(row["X_AM"])) 
        X_Iclr              = np.array(my_eval(row["X_Iclr"])) 
        X_k_bar             = np.array(my_eval(row["X_k_bar"])) 
        X_condition         = np.array(my_eval(row["X_condition"])) 

        GT_cos_zna          = np.array(my_eval(row["GT_cos_zna"]))  
        #GT_am               = np.array(my_eval(row["GT_AM"])) 
        GT_Iclr             = np.array(my_eval(row["GT_Iclr"])) 
        GT_k_bar            = np.array(my_eval(row["GT_k_bar"])) 
        GT_condition        = np.array(my_eval(row["GT_condition"])) 

            
        list_all_files.append(date_time_string)
        dict_list_all_files_per_month[month_].append(list_all_files)

        if os.path.isfile("%s/%s.h5" % (h5_dir, date_time_string)): 
            pbar_new.set_description("[%d/%d] %s.h5 Exist: %s" % (index, len(FileName_list), date_time_string, h5_dir))

        else:

            image_array_stack = []
            gt_image_array_stack = []

            for image_path_name in input_image_names: 

                image_array = iio.imread(os.path.join(image_dir,"%02d" % month_, Date_, image_path_name))   
                if im_resize: 
                    shape              = image_array.shape
                    nr, nc             = shape[0], shape[1]
                    img_pil            = Image.fromarray(image_array) 
                    shrinkFactor       = new_size /min([nr,nc])  
                    img_pil            = img_pil.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
                    image_array_pil    = np.array(img_pil) 
                    image_array_trans  = image_array_pil   
                else:
                    image_array_trans = image_array 
 
                image_array_stack.append(image_array_trans)

            image_array_stack = np.stack(image_array_stack, axis=0)    

            for gt_image_name in gt_image_names:

                gt_image_array    = iio.imread(os.path.join(image_dir,"%02d" % month_, Date_,  gt_image_name)) 

                if im_resize:  
                    shape               = gt_image_array.shape 
                    nr, nc              = shape[0], shape[1]
                    shrinkFactor        = new_size /min([nr,nc])  
                    img_pil_gt          = Image.fromarray(gt_image_array)
                    img_pil_gt          = img_pil_gt.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
                    gt_image_array_pil  = np.array(img_pil_gt) 
                    gt_image_array_trans = gt_image_array_pil
                
                else:
                    gt_image_array_trans  = gt_image_array

                
                gt_image_array_stack.append(gt_image_array_trans)
            
            gt_image_array_stack = np.stack(gt_image_array_stack, axis=0) 

            hf = h5py.File("%s/%s.h5" % (h5_dir, date_time_string), 'w') 
            hf.create_dataset('X', data=image_array_stack)
            hf.create_dataset('Y', data=gt_image_array_stack) 
            hf.create_dataset('X_DateTime_Array', data=X_DateTime_Array)  
            hf.create_dataset('Y_DateTimeArray',  data=GT_DateTimeArray) 

             
            hf.create_dataset('X_Irr', data=X_Irrad)   
            hf.create_dataset('Y_Irr', data=GT_Irrad)  

            hf.create_dataset('X_cos_zna', data=X_cos_zna)  
            #hf.create_dataset('X_am',      data=X_am)  
            hf.create_dataset('X_Iclr',    data=X_Iclr)  
            hf.create_dataset('X_k_bar',   data=X_k_bar) 
            hf.create_dataset('X_condition', data=X_condition) 

            hf.create_dataset('GT_cos_zna', data=GT_cos_zna)  
            #hf.create_dataset('GT_am',      data=GT_am)  
            hf.create_dataset('GT_Iclr',    data=GT_Iclr)  
            hf.create_dataset('GT_k_bar',   data=GT_k_bar) 
            hf.create_dataset('GT_condition', data=GT_condition) 

            hf.close()   
            pbar_new.set_description("[%d/%d] %s.h5 Folder: %s" % (index, len(FileName_list), date_time_string, h5_dir))
# %%

def get_sunny_df(configs): 
    csv_data_temp   = pd.read_csv(os.path.join(configs["Datasets"]["data_folder"][0], configs["Datasets"]["csv_file"][0]), usecols = ['Datetime', 'I', 'Iclr']) 
    csv_data_temp["Datetime"] = pd.to_datetime(csv_data_temp["Datetime"])

    effifan_csv_data_temp   = pd.read_csv(os.path.join(configs["Datasets"]["data_folder"][0], configs["Datasets"]["csv_detect_file"][0]), usecols = ['Datetime', 'daily_cloud_fraction_index' ]) 
    effifan_csv_data_temp["Datetime"] = pd.to_datetime(effifan_csv_data_temp["Datetime"])
    effifan_csv_data_temp["daily_cloud_fraction_index"] = effifan_csv_data_temp["daily_cloud_fraction_index"]/100
    effifan_csv_data_temp = effifan_csv_data_temp.rename(columns={"daily_cloud_fraction_index":"corrected_cloudcover"})

    sunny_pickle_files = glob.glob(os.path.join(configs["Datasets"]["sunny_path"][0], "month_*-h5img.pickle")) 
    sunny_pickle_files.sort() 
    
    for sunny_pickle_index, sunny_pickle_file in enumerate(sunny_pickle_files):
        sunny_pickle_all = pd.read_pickle(sunny_pickle_file) 

        if "h5img_x" in sunny_pickle_all.keys():
            sunny_pickle_temp = sunny_pickle_all[["LocalDatetime", "h5img_x"]]
            sunny_pickle_temp = sunny_pickle_temp.rename(columns={"LocalDatetime":"Datetime", "h5img_x":"h5img"})
        else:
            sunny_pickle_temp = sunny_pickle_all[["LocalDatetime", "h5img"]]
            sunny_pickle_temp = sunny_pickle_temp.rename(columns={"LocalDatetime":"Datetime"})

        if sunny_pickle_index == 0:
            sunny_pickle = sunny_pickle_temp
        else:
            sunny_pickle = pd.concat([sunny_pickle, sunny_pickle_temp], ignore_index=True) 

    csv_data_temp = pd.merge(csv_data_temp, effifan_csv_data_temp, on="Datetime", how="inner") 
    merged_data   = pd.merge(sunny_pickle, csv_data_temp, on="Datetime", how="inner") 

    return merged_data






