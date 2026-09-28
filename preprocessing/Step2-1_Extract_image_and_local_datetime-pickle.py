from sirta_utils.making_CUEE_h5 import ImageFiles_to_H5files, get_cpu_ram_usage 
from sirta_utils.tools import process_images
from PIL import Image
import numpy as np
import pandas as pd
import imageio.v3 as iio
import pdb
import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt


# Usage: python Step2-1_Extract_image_and_local_datetime-pickle.py [year] [imsize]
YEAR      = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
IMSIZE    = int(sys.argv[2]) if len(sys.argv) > 2 else 64
file2h5   = ImageFiles_to_H5files(configs="configs/SIRTA_with_irr_setting_%04d_%d.yaml" % (YEAR, IMSIZE), note_run_date="%04d" % YEAR)
merged_df = file2h5.extract2list()    

merged_df.to_csv(os.path.join(file2h5.folder_name_listing, file2h5.csv_data))

if file2h5.new_size is not None:
    im_resize = True
image_dir = file2h5.image_dir
new_size  = file2h5.new_size 
 
image_and_localtime_list = [] 

pbar = tqdm(merged_df.iterrows(), total=merged_df.shape[0])   


radius      = 300
img_width   = 1024
img_height  = 768
center_x    = img_width // 2
center_y    = img_height // 2
Y, X = np.ogrid[:img_height, :img_width]
dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
mask = dist_from_center <= radius

example_dir = "Step2_Extract_image_and_local_datetime-pickle-%04d"  % YEAR
os.makedirs(example_dir, exist_ok=True)

for index, row in pbar:
    datetime_   = row["Datetime"]
    datetimestr = row["Datetimestr"]
    filename    = row["Name"]

    filename_underexposure    = row["Name_underexpos"]

    year_  = row["Year"]
    month_ = row["Month"]
    day_   = row["day"]

    gt_image_array    = iio.imread(os.path.join(image_dir,  "%02d" % month_ , "%04d%02d%02d" % (year_, month_, day_) , "%s" % filename)) 
    ux_image_array    = iio.imread(os.path.join(image_dir,  "%02d" % month_ , "%04d%02d%02d" % (year_, month_, day_) , "%s" % filename_underexposure)) 

    if im_resize:  
        
        masked_image_cropped, masked_image = process_images(gt_image_array, mask, radius,  new_size=new_size)
        h5img_ux, h5img_ux_masked_image    = process_images(ux_image_array, mask, radius,  new_size=64)

        local_dict = {"LocalDatetime": datetime_, "LocalDatetimestr": datetimestr,  "h5img":masked_image_cropped, "h5img_ux": h5img_ux}
    
    
        if index %2000 == 0:

            img_pil_gt           = Image.fromarray(gt_image_array)
            masked_image_cropped = Image.fromarray(masked_image_cropped)
            maskedimg_width, maskedimg_height = masked_image_cropped.size

            ux_image_array       = Image.fromarray(ux_image_array)
            h5img_ux_cropped     = Image.fromarray(h5img_ux)
            ux_maskedimg_width, ux_maskedimg_height = h5img_ux_cropped.size

            plt.close("all")
            figure, axes = plt.subplots(2,3, figsize=(20, 10))
            axes[0,0].set_aspect('equal') 
            axes[0,0].imshow(img_pil_gt) 
            axes[0,0].scatter(center_x, center_y, s=10, c="red")
            circle = plt.Circle((center_x, center_y), radius, edgecolor="lime", facecolor='none', linewidth=2)
            axes[0,0].add_patch(circle)
            axes[0,0].set_title("Radius: %d" % radius)


            axes[0,1].set_aspect('equal') 
            axes[0,1].imshow(masked_image) 
            axes[0,1].scatter(center_x, center_y, s=10, c="red") 

            axes[0,2].set_aspect('equal') 
            axes[0,2].imshow(masked_image_cropped) 
            axes[0,2].scatter(maskedimg_width // 2, maskedimg_height//2, s=10, c="red")  
            axes[0,2].set_title("Resize: %dx%d" % (maskedimg_height, maskedimg_width)) 


            axes[1,0].set_aspect('equal') 
            axes[1,0].imshow(ux_image_array) 
            axes[1,0].scatter(center_x, center_y, s=10, c="red")
            circle = plt.Circle((center_x, center_y), radius, edgecolor="lime", facecolor='none', linewidth=2)
            axes[1,0].add_patch(circle)
            axes[1,0].set_title("Radius: %d" % radius)


            axes[1,1].set_aspect('equal') 
            axes[1,1].imshow(h5img_ux_masked_image) 
            axes[1,1].scatter(center_x, center_y, s=10, c="red") 

            axes[1,2].set_aspect('equal') 
            axes[1,2].imshow(h5img_ux_cropped)  
            axes[1,2].scatter(ux_maskedimg_width // 2, ux_maskedimg_height//2, s=10, c="red")  
            axes[1,2].set_title("Resize: %dx%d" % (ux_maskedimg_height, ux_maskedimg_width)) 

            plt.savefig("%s/%d-resized.png" % (example_dir, index))
 
 
    
    else:
        masked_image_cropped, masked_image  = process_images(gt_image_array, mask, radius,  new_size=None)
        h5img_ux, h5img_ux_masked_image    = process_images(ux_image_array, mask, radius,  new_size=64)

        local_dict = {"LocalDatetime": datetime_, "LocalDatetimestr": datetimestr,  "h5img":masked_image_cropped, "h5img_ux": h5img_ux}
    
        if index %2000 == 0:
            img_pil_gt           = Image.fromarray(gt_image_array)
            masked_image_cropped = Image.fromarray(masked_image_cropped)
            maskedimg_width, maskedimg_height = masked_image_cropped.size

            ux_image_array       = Image.fromarray(ux_image_array)
            h5img_ux_cropped     = Image.fromarray(h5img_ux)
            ux_maskedimg_width, ux_maskedimg_height = h5img_ux_cropped.size 
 
            plt.close("all")
            figure, axes = plt.subplots(2,3, figsize=(20, 10))
            axes[0,0].set_aspect('equal') 
            axes[0,0].imshow(img_pil_gt) 
            axes[0,0].scatter(center_x, center_y, s=10, c="red")
            circle = plt.Circle((center_x, center_y), radius, edgecolor="lime", facecolor='none', linewidth=2)
            axes[0,0].add_patch(circle)
            axes[0,0].set_title("Radius: %d" % radius)


            axes[0,1].set_aspect('equal') 
            axes[0,1].imshow(masked_image) 
            axes[0,1].scatter(center_x, center_y, s=10, c="red") 

            axes[0,2].set_aspect('equal') 
            axes[0,2].imshow(masked_image_cropped) 
            axes[0,2].scatter(maskedimg_width // 2, maskedimg_height//2, s=10, c="red")  
            axes[0,2].set_title("Resize: %dx%d" % (maskedimg_height, maskedimg_width)) 


            axes[1,0].set_aspect('equal') 
            axes[1,0].imshow(ux_image_array) 
            axes[1,0].scatter(center_x, center_y, s=10, c="red")
            circle = plt.Circle((center_x, center_y), radius, edgecolor="lime", facecolor='none', linewidth=2)
            axes[1,0].add_patch(circle)
            axes[1,0].set_title("Radius: %d" % radius)


            axes[1,1].set_aspect('equal') 
            axes[1,1].imshow(h5img_ux_masked_image) 
            axes[1,1].scatter(center_x, center_y, s=10, c="red") 

            axes[1,2].set_aspect('equal') 
            axes[1,2].imshow(h5img_ux_cropped)  
            axes[1,2].scatter(ux_maskedimg_width // 2, ux_maskedimg_height//2, s=10, c="red")  
            axes[1,2].set_title("Resize: %dx%d" % (ux_maskedimg_height, ux_maskedimg_width)) 

            plt.savefig("%s/%d-resized.png" % (example_dir, index))
 
 
    image_and_localtime_list.append(local_dict) 

    if index % 2000 == 0:
        cpu_usage, ram_usage = get_cpu_ram_usage() 
        print("@%d CPU: %.2f  RAM: %.2f" % (index, cpu_usage, ram_usage))
        pbar.set_description("Image:%s - Local=%s" % (filename, datetimestr))  
    else:
        pbar.set_description("Image:%s - Local=%s" % (filename, datetimestr))

print("Making pickle...")
print("@%d CPU: %.2f  RAM: %.2f" % (index, cpu_usage, ram_usage))
image_and_localtime_dataframe = pd.DataFrame.from_dict(image_and_localtime_list)
image_and_localtime_dataframe.to_pickle(os.path.join(file2h5.folder_name_listing, file2h5.pickle_data) ) 
 