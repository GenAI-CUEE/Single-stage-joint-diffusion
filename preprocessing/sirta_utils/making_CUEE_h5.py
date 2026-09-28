# %%
import glob
import pdb
import os 
from tqdm import tqdm 
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from datetime import date, datetime, time
import numpy as np
from PIL import Image
import csv 
import yaml
import psutil
from sirta_utils.tools import ( imagelist2statdata, 
                        statdata2selecthours, 
                        merge_with_irradiance_csv, 
                        divide2frames, 
                        image2h5files )

class dict2obj:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value) 

def process_configs(default_configs, config_file):   

    if config_file is not None:   

        with open(config_file) as f: 
            yaml_dict = yaml.safe_load(f)  

        for key, value in yaml_dict.items():
            if key in ["start_date", "stop_date", "start_hour", "end_hour"] and (type(value) is list):
                yaml_dict[key] = tuple(value)

        configs = {**default_configs, **yaml_dict}  
    else:
        configs = default_configs  

    return configs

def save_settings_dict(folder_name_listing, configs_dict, file_name = 'setting.csv'): 

    print("================ save settings ================ ")
    for key, value in configs_dict.items():
        if type(value) is tuple: 
            value  = ','.join(map(str, value))
        print("%s: %s" % (key, value))
    
    print("=============================================== \n")

    with open(os.path.join(folder_name_listing, file_name), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in configs_dict.items():
            writer.writerow([key, value])

def get_cpu_ram_usage():
    """Gets the CPU and RAM usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    ram_usage = memory.percent
    return cpu_usage, ram_usage




class ImageFiles_to_H5files:

    default_configs = {
        "image_dir": "SIRTA_skyimages_unzip/2022",  
        "image_resize": 64, # 480, 512
        "TimeRes": 2,
        "seq_length": 16,
        "predict_length": 15,
        "start_date": (2022,  1,  1),
        "stop_date":  (2022, 12, 31),
        "start_hour": (8, 00),
        "end_hour":   (16, 00),
        "is_h5seq": True, 
        "localize" :  None,
        "image_timezone": "Local",
        "start_collecting_images": True,
        "irradiance_csv_folder": "SIRTA_irradiance_solys2",
        "irradiance_csv_ihp_folder": "SIRTA_irradiance_solys2_IPH",
        "irradiance_csv_file": "2022-cleaned_sky-cond.csv",
        }
    
    def __init__(self, configs=None, note_run_date=None): 
        
        now = datetime.now()

        configs_dict =  process_configs(self.default_configs, configs)   
 

        if note_run_date is None: 
            self.Note_run_date = now.strftime("%Y-%m-%d") 
        else:
            self.Note_run_date = note_run_date

        configs = dict2obj(configs_dict)  
        
        configs_dict["Note_run_date"] = self.Note_run_date
        
        self.all_images = glob.glob("%s/*/**/***_01.jpg" % configs.image_dir) 
        self.all_images_ux = glob.glob("%s/*/**/***_03.jpg" % configs.image_dir)   
 
        self.start_collecting_images = configs.start_collecting_images 
 
        if configs.image_resize is not None: 
            self.new_size = configs.image_resize   
            self.im_resize = "%d" % self.new_size 
        else:
            self.im_resize = "%dx%d" % (1920,1080) 
        
        self.image_dir  = configs.image_dir
        self.TimeRes    = configs.TimeRes
        self.seq_length = configs.seq_length  
        self.predict_length = configs.predict_length  
        self.start_date     = configs.start_date 
        self.stop_date      = configs.stop_date 
        self.start_hour     = configs.start_hour
        self.end_hour       = configs.end_hour
        self.irradiance_csv_folder     = configs.irradiance_csv_folder
        self.irradiance_csv_ihp_folder = configs.irradiance_csv_ihp_folder
        self.irradiance_csv_file       = configs.irradiance_csv_file

        start_hour_text     = "%02d:%02d" % (self.start_hour[0], self.start_hour[1])
        end_hour_text       = "%02d:%02d" % (self.end_hour[0], self.end_hour[1])

        self.localize = configs.localize
        self.image_timezone = configs.image_timezone
        self.is_h5seq = configs.is_h5seq


        if self.is_h5seq:

            self.irradiance_csv  = pd.read_csv("%s" % os.path.join(configs.irradiance_csv_ihp_folder, configs.irradiance_csv_file ), usecols=['Datetime', 'I', 'P', 'H', 'cos_zna', 'Iclr', 'k_bar', 'condition'])   
            self.folder_name_listing = "h5files-IRR-Frame-%dx%d-to-%dx%d-Mins_IMS-%s-%s-%s-%s" % (self.TimeRes, self.seq_length, self.TimeRes, self.predict_length, self.im_resize, start_hour_text, end_hour_text, self.Note_run_date)
    
        else:
            self.irradiance_csv = pd.read_csv("%s" % os.path.join(configs.irradiance_csv_ihp_folder, configs.irradiance_csv_file ), usecols=['Datetime', 'I', 'Diff', 'P', 'H', 'C', 'cos_zna', 'Iclr',  "Idhi",  'k_bar', 'condition'])   
            self.folder_name_listing = configs.output_path_image_and_datetime

        os.makedirs(self.folder_name_listing , exist_ok=True) 
        
        configs_dict["Note_foldername"] = self.folder_name_listing

        file_name = "making-h5-setting.csv"
        configs_dict["Note_setting_filename"] = file_name
            
        self.h5_dir = os.path.join(self.folder_name_listing, "h5files") 
        configs_dict["h5_dir"] = self.h5_dir 

        os.makedirs(self.folder_name_listing, exist_ok=True)
      
        if self.is_h5seq:
            os.makedirs(self.h5_dir, exist_ok=True) 
        else:
            configs_dict["pickle_data"] = configs.pickle_data
            configs_dict["csv_data"] = configs.csv_data
            self.pickle_data = configs.pickle_data
            self.csv_data    = configs.csv_data

        save_settings_dict(self.folder_name_listing, configs_dict, file_name=file_name)
        

    def  _imagelist2statdata(self):

        print("\n 1. imagelist2statdata ...===========================================================" )
        print("START Date: %04d-%02d-%02d STOP Date: %04d-%02d-%02d" % (self.start_date[0],self.start_date[1], self.start_date[2], self.stop_date[0], self.stop_date[1], self.stop_date[2]))
        data_frame = imagelist2statdata(self.all_images, start_date = self.start_date, stop_date = self.stop_date, localize_datetime=self.localize, image_timezone=self.image_timezone)

        print(data_frame.iloc[:10])

        return data_frame 
 
    def _statdata2selecthours(self, data_frame):
        # Choosing samples between 8:00 - 16:00

        print("\n 2. statdata2selecthours ......===========================================================")
        print("START Hour: %02d:%02d:00 STOP Hour: %02d:%02d:00" % (self.start_hour[0], self.start_hour[1], self.end_hour[0], self.end_hour[1]))
        
        dataframe_noontime = statdata2selecthours(data_frame, start_hour = self.start_hour,  end_hour = self.end_hour )

        return dataframe_noontime

 
    def _merge_with_irradiance_csv(self, data_frame_noon):
 
        print("\n 2.5 merge_with_irradiance_csv : %s...====================================================" % os.path.join(self.irradiance_csv_folder, self.irradiance_csv_file ))
        merged_df = merge_with_irradiance_csv(data_frame_noon, self.irradiance_csv, self.is_h5seq) 
        
         
        return merged_df

    def _divide2frames(self, merged_df ):
        
        print("\n 3. divide2frames :  ... ==================================================================="  )
 
        if os.path.isfile(os.path.join(self.folder_name_listing,'FileName_list_dataframe.csv')):
            FileName_list = pd.read_csv(os.path.join(self.folder_name_listing,'FileName_list_dataframe.csv'))  

            Month_list_file = open(os.path.join(self.folder_name_listing,'Month_list.txt'), "r") 
            month_list_data = Month_list_file.read()
            Month_list_str  = month_list_data.split("\n") 
            Month_list_file.close() 
 
            Month_list = [int(month_) for month_ in Month_list_str]

            Year_list_file = open(os.path.join(self.folder_name_listing,'Year_list.txt'), "r") 
            year_list_data = Year_list_file.read()
            Year_list_str  = year_list_data.split("\n") 
            Year_list_file.close() 
 
            Year_list = [int(year_) for year_ in Year_list_str] 

            FileName_list.keys() 

        else:
             
            FileName_list, Month_list, Year_list = divide2frames(merged_df, self.TimeRes, self.seq_length, self.predict_length )   
            FileName_list.to_csv( os.path.join(self.folder_name_listing,'FileName_list_dataframe.csv'))  
                 
            file1 = open(os.path.join(self.folder_name_listing,'Month_list.txt'),'w')
            last_line = len(Month_list) - 1
            for num_line, line in enumerate(Month_list):
                if num_line == last_line:
                    file1.write("%s" % line)
                else:
                    file1.write("%s\n" % line)
            file1.close()

            file2 = open(os.path.join(self.folder_name_listing,'Year_list.txt'),'w')
            last_line = len(Year_list) - 1
            for num_line,  line in enumerate(Year_list):
                if num_line == last_line:
                    file2.write("%s" % line)
                else:
                    file2.write("%s\n" % line)
            file2.close()


        return FileName_list, Month_list, Year_list
    
    def _image2h5files(self, FileName_list, Month_list, Year_list ):
        
        print("\n 4. image2h5files :  ... ==================================================================="  )
        if self.start_collecting_images:
            print("Final step !.... image2h5files :  ..."  ) 
            image2h5files(FileName_list, Month_list, Year_list, self.h5_dir, self.image_dir, self.im_resize, self.new_size )

    def run(self):

        # 1. Putting each image into dataframe:
        # | "Name"     |      "Datetime"         |  "Date" |  "Date_str": '%04d-%02d-%02d'|  "Minutes" |  "Time" | "Year" | "Month" | "day" | "hour" | "min" |
        #  ----------------------------------------------------------------------------------------------------------------------------------------------
        # | image_name |  datetime.fromisoformat |  .... 
        #  
        data_frame = self._imagelist2statdata()
 
        # 2. Selecting hours from 6:30 - 17:15  
        data_frame_noon = self._statdata2selecthours(data_frame)
 
        
        if self.is_h5seq:
            # 3. Merge with dataframe from irradiance file 
            merged_df = self._merge_with_irradiance_csv(data_frame_noon)

            # 4. Remove incomplete data and stacking data together ...
            FileName_list, Month_list, Year_list = self._divide2frames(merged_df) 

        else: 
            FileName_list, Month_list, Year_list = self._divide2frames(data_frame_noon)
 
        self._image2h5files(FileName_list, Month_list, Year_list ) 

    def extract2list(self):
        data_frame      = self._imagelist2statdata()
        data_frame_noon = self._statdata2selecthours(data_frame)
        merged_df       = self._merge_with_irradiance_csv(data_frame_noon)

        return merged_df


if __name__ == "__main__":

    file2h5 = ImageFiles_to_H5files(configs="configs/cuee_without_irr_setting.yaml")
    file2h5.run() 