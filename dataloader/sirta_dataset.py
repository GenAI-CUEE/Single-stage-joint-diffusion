import yaml

import pdb
import os
import re

import sys
import pathlib
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from   tqdm import tqdm 
import numpy as np 

import torch 
import torch.utils.data as data
from torch.utils.data import DataLoader  
from datetime import datetime  

from termcolor import colored

from dataloader.sirta_dataset_utils import resize_image, get_input_file_list, get_datetime_from_numpy, write_to_csv
import pandas as pd

# sys.path.insert(1, os.path.join(os.getcwd(), 'Sky_StableDiffusion_Master'))     
# from Sky_StableDiffusion_Master.modules.conditionalDDPM import ConditionalDDMP

class SeqDataset_SIRTA(data.Dataset):
    configs={ }
    def __init__(self, configs, device, mode="test", dataset="SIRTA", attribute_list = ["label", "feature"], SkyGPT_model=None) :
        super(SeqDataset_SIRTA, self).__init__() 

        # loading data
        self.device = device
        self.name   = dataset  
        self.data_folder          = configs["Data"]["data_folder"]   

        csv_label                 = configs["Data"]["csv_label"]
        use_cols_csv_label        = configs["Data"]["use_cols_csv_label"]
        pickle_image              = configs["Data"]["pickle_image"]
        use_cols_pickle_image     = configs["Data"]["use_cols_pickle_image"]
        time_channel_coupling_enable = configs["Data"]["time_channel_coupling_enable"] # True if you want to couple the time and image channels together.

        image_size      = configs["ImageSequence"]["image_size"]
        image_resize    = configs["ImageSequence"]["image_resize"]
        seq_length      = configs["ImageSequence"]["seq_length"]
        pred_length     = configs["ImageSequence"]["pred_length"]
        image_list_file = { "test"  : os.path.join(self.data_folder, configs["Data"]["test_list"]), 
                            "valid" : os.path.join(self.data_folder, configs["Data"]["valid_list"]), 
                            "train" : os.path.join(self.data_folder, configs["Data"]["train_list"]) }
        
        if not(image_size == image_resize):
            self.image_resize = image_resize
        else:
            self.image_resize = False

        self.use_cols_csv_label        = use_cols_csv_label
        self.use_cols_pickle_image     = use_cols_pickle_image

        self.csv_label             = csv_label
        self.pickle_image          = pickle_image
        self.image_size            = image_size
        self.seq_length            = seq_length
        self.pred_length           = pred_length

        self.time_channel_coupling_enable   = time_channel_coupling_enable

        self.image_list_file = image_list_file

        # Cross-year holdout (train/valid on Data.years_train_valid, test
        # on Data.years_test) -- OFF by default, so every config that doesn't set
        # this key (i.e. every config today) falls straight into the single-year
        # path below, byte-for-byte unchanged.
        self.cross_year_holdout = configs["Data"].get("cross_year_holdout", False)

        if not self.cross_year_holdout:

            year_match = re.search(r"_(\d{4})\.(csv|pickle)$", os.path.basename(self.csv_label))
            if not year_match:
                raise ValueError("csv_label %r does not end in _YYYY.csv -- can't determine the year"
                                  % self.csv_label)
            self.year   = year_match.group(1)

            print("YEAR : %s ...... " % (self.year))

            print("Reading csv_label: [%s]" % os.path.join(self.data_folder, csv_label))
            all_data = pd.read_csv( os.path.join(self.data_folder, csv_label),
                                                usecols = use_cols_csv_label,
                                                index_col = False )
            all_data["Datetime"] = pd.to_datetime(all_data["Datetime"])
            all_data = all_data.drop_duplicates(subset = 'Datetime')

            if ("image" in attribute_list)  :

                print("Reading pickle_image image: [%s]" % os.path.join(self.data_folder, pickle_image))
                pickle_data = pd.read_pickle( os.path.join(self.data_folder, pickle_image) )
                pickle_data["Datetime"] = pd.to_datetime(pickle_data["LocalDatetime"])
                pickle_data = pickle_data.drop(columns = ["LocalDatetime"])
                pickle_data = pickle_data.drop_duplicates(subset = 'Datetime')
                pickle_data = pickle_data[use_cols_pickle_image]

                all_data = pd.merge(all_data, pickle_data, on = 'Datetime', how = 'inner')

            all_data = all_data.drop_duplicates(subset='Datetime')
            print("Finish loading data")

            self.n_samples   = len(all_data) - seq_length - pred_length - 1
            print("All data : %d" % self.n_samples)

            os.makedirs(self.data_folder, exist_ok=True)

            dict_ref_label   = self.sorting_data(all_data, attribute = "label")
            dict_ref_feature = self.sorting_data(all_data, attribute = "feature")
            dict_ref_image   = self.sorting_data(all_data, attribute = "image")

            intersection_timestamp = list(set(dict_ref_label["timestamp"]) & set(dict_ref_feature["timestamp"] ))
            intersection_timestamp = list(set(intersection_timestamp) & set(dict_ref_image["timestamp"] ))

            dict_ref_label   = pd.DataFrame.from_dict(dict_ref_label)
            dict_ref_feature = pd.DataFrame.from_dict(dict_ref_feature)
            dict_ref_image   = pd.DataFrame.from_dict(dict_ref_image)

            dict_ref_label   = dict_ref_label[dict_ref_label["timestamp"].isin(intersection_timestamp)]
            dict_ref_feature = dict_ref_feature[dict_ref_feature["timestamp"].isin(intersection_timestamp)]
            dict_ref_image   = dict_ref_image[dict_ref_image["timestamp"].isin(intersection_timestamp)]

            datetime_list = get_input_file_list(self.image_list_file[mode])

            dict_mode_label = dict_ref_label[dict_ref_label["timestamp"].isin(datetime_list)]
            dict_mode_feature = dict_ref_feature[dict_ref_feature["timestamp"].isin(datetime_list)]
            dict_mode_image = dict_ref_image[dict_ref_image["timestamp"].isin(datetime_list)]

            print(colored("[Mode = %s] Num. samples : %d [out of %d]" % (mode, len(dict_mode_label), self.n_samples) , 'cyan'))

            self.n_samples = len(dict_mode_label)

        else:

            # --- Cross-year path ------------------------------------
            # Same load -> sorting_data -> intersection -> mode-list-filter shape
            # as the single-year block above, run once per year (via
            # _load_one_year_frames) and concatenated at the dict_mode_*
            # (already-sequenced) level -- never concatenating raw frames across
            # years before sequencing, so no sequence can span a year boundary
            # (Gate 3).
            if mode in ("train", "valid"):
                years = configs["Data"]["years_train_valid"]
                list_key = "train_list_cross_year" if mode == "train" else "valid_list_cross_year"
                mode_list_path = os.path.join(configs["Data"]["cross_year_list_folder"],
                                               configs["Data"][list_key])
            elif mode == "test":
                years = configs["Data"]["years_test"]
                mode_list_path = self.image_list_file[mode]
            else:
                raise ValueError("cross_year_holdout: unknown mode '%s'" % mode)

            self.year = "-".join(str(y) for y in years)
            print("YEARS (cross_year_holdout) : %s ...... " % (self.year))

            mode_datetime_list = get_input_file_list(mode_list_path)

            year_frames = {"label": [], "feature": [], "image": []}
            year_data_folders = []
            for year in years:
                frames, year_data_folder = self._load_one_year_frames(
                    year, attribute_list, mode_datetime_list,
                    use_cols_csv_label, use_cols_pickle_image)
                year_data_folders.append(year_data_folder)
                for key in ("label", "feature", "image"):
                    year_frames[key].append(frames[key])

            self.data_folder = ",".join(year_data_folders)

            dict_mode_label   = pd.concat(year_frames["label"],   ignore_index=True)
            dict_mode_feature = pd.concat(year_frames["feature"], ignore_index=True)
            dict_mode_image   = pd.concat(year_frames["image"],   ignore_index=True)

            print(colored("[Mode = %s] [Years = %s] Num. samples : %d" %
                           (mode, self.year, len(dict_mode_label)), 'cyan'))

            self.n_samples = len(dict_mode_label)

        # Find the intersect of the reference x and y
        self.attribute_list = []
        self.data = {}

        self.set_inputs_to_get_item(dict_mode_label,      attribute = "ref")
        self.set_inputs_to_get_item(dict_mode_label,      attribute = "label")
        self.set_inputs_to_get_item(dict_mode_feature,    attribute = "feature")
        self.set_inputs_to_get_item(dict_mode_image,      attribute = "image")

        # convert_image2vqvae (precompute-and-cache-in-RAM the whole split's
        # frozen-VQVAE encodings) was removed 2026-08-10 -- it OOM-killed Phase-2
        # training (the 12-frame-clip cache is ~57GB for this dataset's training
        # split). models/SkyGPT.py::train() now encodes on the fly per batch instead
        # (same as evaluation() already did), so nothing calls this path anymore.

        # initialized n_sample (this value will be updated twice)


    def reset_data_for_get_item(self):
        self.attribute_list = []
        self.data = {}
 
    def set_inputs_to_get_item(self, data_dict, attribute):  
        # attribute 
        self.attribute_list.append(attribute)  
        self.data["%s_x" % attribute]   = data_dict["%s_x" % attribute].tolist() 
        self.data["%s_y" % attribute]   = data_dict["%s_y" % attribute].tolist()    
 
 

    def _load_one_year_frames(self, year, attribute_list, mode_datetime_list,
                               use_cols_csv_label, use_cols_pickle_image):
        """Cross-year holdout helper: load + merge + sequence + mode-filter a
        single year, mirroring the single-year block in __init__ exactly so that
        calling this once per year and concatenating the returned
        (already-sequenced) frames is equivalent to running that block once per
        year."""

        data_folder  = "sirta_%d_image64-localtime_ICHP-daytime-dhi" % year
        csv_label    = "sirta_data_%d.csv" % year
        pickle_image = "sirta_image_%d.pickle" % year

        os.makedirs(data_folder, exist_ok=True)

        print("YEAR : %d ...... " % (year))

        print("Reading csv_label: [%s]" % os.path.join(data_folder, csv_label))
        all_data = pd.read_csv(os.path.join(data_folder, csv_label),
                                usecols=use_cols_csv_label, index_col=False)
        all_data["Datetime"] = pd.to_datetime(all_data["Datetime"])
        all_data = all_data.drop_duplicates(subset='Datetime')

        if ("image" in attribute_list):
            print("Reading pickle_image image: [%s]" % os.path.join(data_folder, pickle_image))
            pickle_data = pd.read_pickle(os.path.join(data_folder, pickle_image))
            pickle_data["Datetime"] = pd.to_datetime(pickle_data["LocalDatetime"])
            pickle_data = pickle_data.drop(columns=["LocalDatetime"])
            pickle_data = pickle_data.drop_duplicates(subset='Datetime')
            pickle_data = pickle_data[use_cols_pickle_image]
            all_data = pd.merge(all_data, pickle_data, on='Datetime', how='inner')

        all_data = all_data.drop_duplicates(subset='Datetime')
        print("Finish loading data")

        dict_ref_label   = self.sorting_data(all_data, attribute="label")
        dict_ref_feature = self.sorting_data(all_data, attribute="feature")
        dict_ref_image   = self.sorting_data(all_data, attribute="image")

        intersection_timestamp = list(set(dict_ref_label["timestamp"]) & set(dict_ref_feature["timestamp"]))
        intersection_timestamp = list(set(intersection_timestamp) & set(dict_ref_image["timestamp"]))

        dict_ref_label   = pd.DataFrame.from_dict(dict_ref_label)
        dict_ref_feature = pd.DataFrame.from_dict(dict_ref_feature)
        dict_ref_image   = pd.DataFrame.from_dict(dict_ref_image)

        dict_ref_label   = dict_ref_label[dict_ref_label["timestamp"].isin(intersection_timestamp)]
        dict_ref_feature = dict_ref_feature[dict_ref_feature["timestamp"].isin(intersection_timestamp)]
        dict_ref_image   = dict_ref_image[dict_ref_image["timestamp"].isin(intersection_timestamp)]

        dict_mode_label   = dict_ref_label[dict_ref_label["timestamp"].isin(mode_datetime_list)]
        dict_mode_feature = dict_ref_feature[dict_ref_feature["timestamp"].isin(mode_datetime_list)]
        dict_mode_image   = dict_ref_image[dict_ref_image["timestamp"].isin(mode_datetime_list)]

        print(colored("[Year = %d] Num. samples : %d" % (year, len(dict_mode_label)), 'cyan'))

        frames = {
            "label": dict_mode_label, "feature": dict_mode_feature, "image": dict_mode_image,
        }
        return frames, data_folder

    def sorting_data(self, all_data,  attribute : str = "label"):

        data_folder_ = self.data_folder   
        dict_ref = self.making_seqdata_one_year(all_data, attribute = attribute) 
       

        ref_y           = dict_ref["ref_y"]  
        ref_first_y     = [ref_y_[0,:] for ref_y_ in ref_y] 
        datetime_y_list = get_datetime_from_numpy(ref_first_y) 
        
        dict_ref["timestamp"] = datetime_y_list   
         
        return dict_ref
 

    def making_seqdata_one_year(self, all_data, attribute : str = "label"):  
        
        reference      = self.convert_df_to_ref(all_data)
        print(f"reference shape:  {reference.shape}")  

        if attribute == "label":
            data          = self.convert_df_irradiance_to_list(all_data)
            print(f"label shape:  {data.shape}")   
        
        elif attribute == "feature":
            data        = self.convert_df_features_to_list(all_data)
            print(f"feature shape: {data.shape}")   
        
        elif attribute == "image":
            data          = self.convert_df_image_to_list(all_data)
            print(f"image shape: {data.shape}")

        ref_x, ref_y, data_x, data_y = self.making_data2sequence(reference, data)
 

        key_x = "%s_x" % attribute
        key_y = "%s_y" % attribute

        dict_ref = {"ref_x": ref_x, "ref_y": ref_y, key_x : data_x, key_y : data_y} 

        return dict_ref  
    
    
    def making_data2sequence(self, reference, input_data):
        num_data = reference.shape[0] - self.seq_length + self.pred_length - 1   

        pbar = tqdm(range(num_data))
 

        ref_x = []
        ref_y = [] 

        data_x = []
        data_y = []
 

        for ind in pbar:

            start_x = ind
            end_x   = ind + self.seq_length 
            start_y = ind + self.seq_length 
            end_y   = ind + self.seq_length + self.pred_length
 

            ref_x_ = reference[ start_x : end_x, :]
            ref_y_ = reference[ start_y : end_y, :] 
 
            check_day_x = [ref_x_[i, 2] == ref_x_[0, 2] for i in range(ref_x_.shape[0])]
            check_day_y = [ref_y_[i, 2] == ref_y_[0, 2] for i in range(ref_y_.shape[0])]

 
            check_day_  = check_day_x + check_day_y

            if all(check_day_) == True and (ref_x_.shape[0] == self.seq_length) and (ref_y_.shape[0] == self.pred_length):  
                
                if np.ndim(input_data) == 1:
                    data_x_ = input_data[start_x : end_x]
                    data_y_ = input_data[start_y : end_y]

                elif np.ndim(input_data) == 2:
                    data_x_ = input_data[start_x : end_x, : ]
                    data_y_ = input_data[start_y : end_y, : ] 

                elif np.ndim(input_data) == 3:
                    data_x_ = input_data[start_x : end_x, : , :  ]
                    data_y_ = input_data[start_y : end_y, : , :  ]  

                elif np.ndim(input_data) == 4:
                    # Assume that input_data with 4 dimensions is an image of size B x H x W x 3 
                    data_x_ = input_data[start_x : end_x, : , : , : ]
                    data_y_ = input_data[start_y : end_y, : , : , : ] 

                    if self.time_channel_coupling_enable:
                        data_x_ = np.transpose(data_x_, axes=[0, 3, 1, 2]).reshape(self.seq_length * 3, 64, 64)  # T x H x W x 3 >> T x 3 x H x W  >> T*3 x H x W
                        data_y_ = np.transpose(data_y_, axes=[0, 3, 1, 2]).reshape(self.pred_length * 3, 64, 64) # T x H x W x 3 >> T x 3 x H x W  >> T*3 x H x W
                    else:
                        data_x_ = np.transpose(data_x_, axes=[0, 3, 1, 2])  # T x H x W x 3 >> T x 3 x H x W  
                        data_y_ = np.transpose(data_y_, axes=[0, 3, 1, 2])  # T x H x W x 3 >> T x 3 x H x W  
  
                ref_x.append(ref_x_)
                ref_y.append(ref_y_)
                data_x.append(data_x_)
                data_y.append(data_y_) 

                pbar.set_description("[%d] [all = %s] seq-x %d - %d | seq-y %d - %d" % (ind, all(check_day_), start_x, end_x, start_y, end_y))
            else: 
                pbar.set_description("[%d] [all = %s] " % (ind, all(check_day_))) 
 
        return  ref_x, ref_y, data_x, data_y 
            
    
    def convert_df_to_ref(self, all_data):
        hour   = all_data['Datetime'].dt.hour.tolist()
        minute = all_data['Datetime'].dt.minute.tolist()
        day    = all_data['Datetime'].dt.day.tolist()
        month  = all_data['Datetime'].dt.month.tolist()
        year   = all_data['Datetime'].dt.year.tolist()
        
        reference = []
        for h_, m_, day_, month_, year_ in zip(hour, minute, day, month, year): 
            reference.append(np.array([h_, m_, day_, month_, year_], dtype="uint16"))    

        reference = np.stack(reference)
        return reference

    def convert_df_features_to_list(self, all_data):
        hour        = all_data['Datetime'].dt.hour.tolist()
        minute      = all_data['Datetime'].dt.minute.tolist()
        day_of_year = all_data['Datetime'].dt.dayofyear.tolist()  
        Iclr        = all_data['Iclr'].tolist()

        feature     = []  
        for h_, m_, doy_, iclr_ in zip(hour, minute, day_of_year, Iclr):
            feature.append(np.array([h_, m_, doy_, iclr_], dtype="float32"))   
        feature = np.stack(feature) 
        return feature 
    
    def convert_df_irradiance_to_list(self, all_data): 
        label = all_data['I'].values 
        label = label.reshape(-1,1) 
        return  label


    def convert_df_image_to_list(self, all_data):
        image_list = all_data['h5img'].tolist()  

        if self.image_resize: 
            prev_image_list = image_list
            image_list = [] 
            for image_ in prev_image_list: 
                scimage = image_.astype(np.uint8)  
                scimage = resize_image(scimage, new_size=self.image_resize)   
                image_list.append(scimage) 
        image_list = np.stack(image_list)
        return image_list 



    def get_input_file_list(self, data_folder, image_list_file=None):

        my_file      = open(os.path.join(data_folder, image_list_file), "r")
        textdata     = my_file.read()
        my_file.close()
 
        datetime_list = textdata.split("\n") 
 
        date_obj_list = []
        for date_str_ in datetime_list: 
            if date_str_ != '':
                datetime_ = datetime.strptime(date_str_, '%Y-%m-%d %H:%M:%S')
                date_obj_list.append(datetime_)  

        return  date_obj_list 
    
    def __getitem__(self, idx):

        # We only use this getitem for checking the result ... 
        # We did not use it for training the model. 
        outdict = {}
 
        for attribute in self.attribute_list:
            outdict["%s_x" % attribute] = self.data["%s_x" % attribute][idx]
            outdict["%s_y" % attribute] = self.data["%s_y" % attribute][idx]
        return outdict

    def __len__(self):
        return self.n_samples
