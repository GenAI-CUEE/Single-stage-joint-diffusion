import yaml
import numpy as np
from math import log10, sqrt 

import pdb
import glob
import math
import os
import random

import pandas as pd
from   tqdm import tqdm 
import matplotlib.pyplot as plt 
 
from torchvision import transforms 

from PIL import Image 
import pickle 
from datetime import datetime, timedelta
import csv

def write_to_csv(rows, csv_file_path): 
    # Opening the CSV file in read and
    # write mode using the open() module
    with open(csv_file_path, 'r+', newline='') as file:

        # creating the csv writer
        file_write = csv.writer(file)

        # Iterating over all the data in the rows 
        # variable
        for val in rows:
          
            # writing the data in csv file
            file_write.writerow(val)


def get_datetime_from_numpy(datetime_array_list):
    total_timestamp = len(datetime_array_list)

    date_obj_list = []
    for date_arr_ in datetime_array_list: 
        date_str_ = "%04d-%02d-%02d %02d:%02d" % (date_arr_[4], date_arr_[3], date_arr_[2], date_arr_[0],  date_arr_[1])
        datetime_ = datetime.strptime(date_str_, '%Y-%m-%d %H:%M') 
        date_obj_list.append(datetime_)  

    return  date_obj_list

def get_input_file_list(image_list_file=None):

    my_file      = open(image_list_file, "r")
    textdata     = my_file.read()
    my_file.close()

    datetime_list = textdata.split("\n") 

    date_obj_list = []
    for date_str_ in datetime_list: 
        if date_str_ != '':
            datetime_ = datetime.strptime(date_str_, '%Y-%m-%d %H:%M:%S')
            date_obj_list.append(datetime_)  

    return  date_obj_list  

def transform_images(image_list, image_size=128): 

    train_transforms = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    ) 
    
    images = []
    pbar = tqdm(image_list, total=len(image_list))
    for image in pbar:  
        images.append(train_transforms(Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")))
        pbar.set_description("image transformation")
    return images


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def crop_center(img, cropx=None, cropy=None):
    y, x, _ = img.shape

    if cropy == None:
        cropy = y
    if cropx == None:
        cropx = x

    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2) 

    return img[starty:starty+cropy, startx:startx+cropx, :] 


def resize_image(image_nparray, new_size=None): 

    if new_size is not None: 

        image_PIL             = Image.fromarray(image_nparray)   
        img_width, img_height = image_PIL.size  
        shrinkFactor          = new_size /min([img_width, img_height])   

        resized_image_PIL = image_PIL.resize((round(img_width*shrinkFactor),round(img_height*shrinkFactor)))
        resized_width, resized_height = resized_image_PIL.size  
        image_nparray     = np.array(resized_image_PIL)  
         
    return image_nparray


def sample_plot(img, context, img_name):
    plt.figure()
    plt.imshow(img)
    plt.title("[%02d-%02d-%03d-%.2f-%.2f-%.2f]" %(context[0], context[1], context[2], context[3], context[4], context[5]))
    plt.savefig("%s.png" % img_name, bbox_inches='tight')