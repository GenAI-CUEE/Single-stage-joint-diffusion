import glob
import pdb
import os  
import h5py 
import numpy as np
from datetime import date, datetime, time
from sirta_utils.making_CUEE_h5 import process_configs as process_yamlconfigs
from sirta_utils.making_CUEE_h5 import save_settings_dict
from sirta_utils.tools import extract_dict_dayslist
import csv
import pandas as pd
import random

class dict2obj:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value) 

def process_csvconfigs(default_configs, config_file=None):

    if config_file is not None:   

        reader = csv.reader(open(config_file, 'r'))
        dict_config = {}
        for row in reader:
            key, value = row
            if key in ["start_date", "stop_date", "start_hour", "end_hour"]:
                dict_config[key] = tuple(value)
            else:
                if key in ["image_resize", "TimeRes", "seq_length", "predict_length"]:
                    dict_config[key] = int(value) 
                elif key in ["percent_training", "percent_valid", "percent_testing" ]:
                    dict_config[key] = float(value) 
                elif key in ["is_Irrad", "start_collecting_images"]:
                    dict_config[key] = bool(value) 
                else:
                    dict_config[key] = value 

        configs = {**default_configs, **dict_config}  

    else:
        configs = default_configs
 
    configs["start_month"] = eval(("").join(dict_config["start_date"]))[1]
    configs["end_month"]   = eval(("").join(dict_config["stop_date"]))[1] 
    
    return configs

def intersection(lst1, lst2): 
    return len(list(set(lst1) & set(lst2)))

def setunion(lst1, lst2, lst3):
    return sorted(list(set(lst1)) + list(set(lst2))  + list(set(lst3)) )

def exclude_from_list(lst_full, lst, total_index): 
    newlist = sorted(list(set(lst_full).difference(set(lst))))
    new_index = [total_index[index]  for index in range(len(lst_full)) if lst_full[index] not in lst] 
    excluded_index = [total_index[index]  for index in range(len(lst_full)) if lst_full[index] in lst] 
    return newlist, np.array(new_index), excluded_index

def read_text2list(filename):
    my_file = open(filename, "r") 
    data = my_file.read()
    data_into_list = data.split("\n") 
    my_file.close() 
    return data_into_list



class Spliting_H5files:

    default_configs = { 
        "percent_training": 0.60,
        "percent_valid":    0.20,
        "percent_testing":  0.20,
        "TimeRes": None, 
        "seq_length": None,
        "predict_length":None,
        "start_month":3,
        "end_month":11,
        "Note_run_date":None,
        "use_extra": None, #"./extras/month09_Testing.txt"
        "Note":None
        }
    
    def __init__(self, h5_mainfolder=None, configs=None): 

        if configs is not None:
            if configs.split(".")[1] == "yaml":

                dict_configs = process_yamlconfigs(self.default_configs, configs)    
            
            elif configs.split(".")[1] == "csv":

                dict_configs = process_csvconfigs(self.default_configs, configs) 

            self.args = dict2obj(dict_configs)   
        else:
            self.args = dict2obj(self.default_configs)
         
 
        self.is_h5seq = eval(self.args.is_h5seq)

        if h5_mainfolder == None:
            self._h5foldername_from_argsettings()
        else:
            self.h5_mainfolder = h5_mainfolder
 

        self.dict_ = {} # To collect extra settings 
        
        self._txtfile_splitting_name()
        self.root = os.getcwd() 
        self.dict_["h5_mainfolder"] = self.h5_mainfolder  
        self.dict_["filelist_destination_root"] = self.h5_mainfolder 
        self.dict_["use_extra"] = self.args.use_extra 

    def _h5foldername_from_argsettings(self):  
 
        if (self.args.TimeRes is None) or (self.args.seq_length is None) or (self.args.predict_length is None):
            return ValueError("You should specify TimeRes, seq_length, and predict_length or manually provide the 'h5_mainfolder'")
        else: 
            if self.is_h5seq:
                self.h5_mainfolder = self.args.Note_foldername
            else:
                self.h5_mainfolder = self.args.output_path_image_and_datetime
 

    def _txtfile_splitting_name(self):

        self.dict_["percent_training"] = self.args.percent_training
        self.dict_["percent_valid"]    = self.args.percent_valid
        self.dict_["percent_testing"]  = self.args.percent_testing
        self.dict_["image_resize"]     = self.args.image_resize  

        if (self.args.TimeRes is None) or (self.args.seq_length is None) or (self.args.predict_length is None):
            setting = "Tr%0.2f-Val%0.2f-Test%0.2f" % (self.args.percent_training, self.args.percent_valid, self.args.percent_testing)
            self.setting = setting.replace(".","p") 
        else: 
            self.dict_["TimeRes"]    = self.args.TimeRes
            self.dict_["seq_length"]  = self.args.seq_length
            self.dict_["predict_length"] = self.args.predict_length
            
            setting = "Tr%0.2f-Val%0.2f-Test%0.2f-Frame-%dx%d-to-%dx%d" % (self.args.percent_training, self.args.percent_valid, self.args.percent_testing, self.args.TimeRes, self.args.seq_length, self.args.TimeRes, self.args.predict_length)
            self.setting = setting.replace(".","p") 
 
        if self.args.Note is not None:
            self.setting = "%s-%s" % (self.setting, self.args.Note)
 
        self.dict_["start_month"]    = self.args.start_month
        self.dict_["end_month"]      = self.args.end_month


    def custom_print(self, message_to_print, log_file='train-test-valid-splitting_log.txt'):
        log_file_path = os.path.join(self.h5_mainfolder, log_file) 
        print(message_to_print)
        with open(log_file_path, 'a') as of:
            of.write(message_to_print + '\n')

    # days_list_per_month, date2files, file2datetime        
    def separate_index_for_testing_valid_traning(self, month_, day_list_per_month, date2files=None, file2datetime=None, num_file_per_month=None):
        
        percent_testing  = self.args.percent_testing
        percent_valid    = self.args.percent_valid

        # ----------------------------------
        # 1. Get total_index (the index of all files in month) and the number of testing and validation  
        file_list_per_month_org = day_list_per_month
        total_index = np.arange(num_file_per_month)  
    
        num_testing       = int(percent_testing*num_file_per_month)
        num_valid         = int(percent_valid*num_file_per_month) 

        # ----------------------------------
        # 2. Randomize testing file index (rand_index_testing) and valid index files (valid_index_list). 
        # Then, from the total_index, substract the testing and validating index to get the training... 
        np.random.permutation(total_index) 
        rand_index_testing     = total_index[:num_testing]   

        # From the total_index, substract the testing to get the training and validating set...
        all_training_index_list  = np.setdiff1d(total_index, rand_index_testing, assume_unique=False)  
        valid_index_list         = total_index[num_testing+1: num_testing+num_valid+1] 
        # Then, from all_training_index_list, substract the validating index to get the training... 
        subtraining_index_list   = np.setdiff1d(all_training_index_list, valid_index_list, assume_unique=False)     
 
        # list(set(rand_index_testing) & set(valid_index_list))
        # pdb.set_trace()
        # list(set(rand_index_testing) & set(subtraining_index_list))
        # pdb.set_trace()
        # list(set(valid_index_list) & set(subtraining_index_list))
        # pdb.set_trace()
        # ----------------------------------  
        # 3.  Use these index (subtraining_index_list, valid_index_list, rand_index_testing) to select the h5 files, and put them into dictionary 
        testing_daylist  = [ day_list_per_month[index] for index in rand_index_testing.tolist()] 
        valid_daylist    = [ day_list_per_month[index] for index in valid_index_list.tolist()]  
        training_daylist = [ day_list_per_month[index] for index in subtraining_index_list.tolist()]  

        # ----------------------------------         
        # 4. turning from dates to files ...
        testing_filelist = []
        if self.is_h5seq:

            for day_ in testing_daylist: 
                files_ = date2files[day_]   
                for files_temp in files_:
                    datetime_ = file2datetime[files_temp]
                    files_and_date = "%s, %s" % (files_temp, datetime_) 
                    testing_filelist.append(files_and_date) 
        else:
            testing_filelist = testing_daylist
 
        valid_filelist = []
        if self.is_h5seq:

            for day_ in valid_daylist: 
                files_ = date2files[day_]   
                for files_temp in files_:
                    datetime_ = file2datetime[files_temp]
                    files_and_date = "%s, %s" % (files_temp, datetime_) 
                    valid_filelist.append(files_and_date) 
        else:
            valid_filelist = valid_daylist

        training_filelist = []
        if self.is_h5seq:
            

            for day_ in training_daylist: 
                files_ = date2files[day_]   
                for files_temp in files_:
                    datetime_ = file2datetime[files_temp]
                    files_and_date = "%s, %s" % (files_temp, datetime_) 
                    training_filelist.append(files_and_date) 

        else:
            training_filelist = training_daylist
            
        return testing_filelist, valid_filelist, training_filelist, file_list_per_month_org


    def dict2text(self, dict_filelist_testing_month,dict_filelist_valid_month, dict_filelist_training_month):
        setting = self.setting 
        h5_mainfolder = self.h5_mainfolder
        
        list_testing = []
        list_validating = []
        list_training = []
 
        self.custom_print("======================== Put into List ==========================================")

        for keys_, values_ in dict_filelist_testing_month.items(): 
            
            list_testing += dict_filelist_testing_month[keys_]

            list_validating += dict_filelist_valid_month[keys_]
            
            list_training += dict_filelist_training_month[keys_]
            self.custom_print("[%s] Testing %d / Valid %d / Training %d" % (keys_, len(dict_filelist_testing_month[keys_]),len(dict_filelist_valid_month[keys_]) ,len(dict_filelist_training_month[keys_])))
        
        total_sample = len(list_testing) + len(list_validating) + len(list_training)
        self.custom_print("[Total: %d] Testing %d / Valid %d / Training %d" % (total_sample, len(list_testing),len(list_validating) ,len(list_training)))
        self.custom_print("[Total: %d] Testing %0.2f / Valid %0.2f / Training %0.2f" % (total_sample, len(list_testing)/total_sample,len(list_validating)/total_sample,len(list_training)/total_sample))


        if self.is_h5seq:
            with open(os.path.join(h5_mainfolder, "Test_%s.txt" % setting)  , 'w') as output:
                for row in list_testing:
                    row = os.path.basename(row)
                    output.write(row + '\n')

            with open(os.path.join(h5_mainfolder, "Valid_%s.txt" % setting)  , 'w') as output:
                for row in list_validating:
                    row = os.path.basename(row)
                    output.write(row + '\n')

            with open(os.path.join(h5_mainfolder, "Train_%s.txt" % setting), 'w') as output:
                for row in list_training:
                    row = os.path.basename(row)
                    output.write(row + '\n')
        else:
             
            with open(os.path.join(h5_mainfolder, "Test_%s.txt" % setting)  , 'w') as output:
                for row in list_testing:  
                    output.write(str(row) + '\n')

            with open(os.path.join(h5_mainfolder, "Valid_%s.txt" % setting)  , 'w') as output:
                for row in list_validating: 
                    output.write(str(row) + '\n')

            with open(os.path.join(h5_mainfolder, "Train_%s.txt" % setting), 'w') as output:
                for row in list_training: 
                    output.write(str(row) + '\n')      

        return list_training, list_validating, list_testing 
    
         
    def extract_file2list(self, textfile, which_key="Datetime"):
        file_testing = open(os.path.join(self.args.use_extra, textfile), 'r') 
        file_list = file_testing.readlines()  

        output_list = []
        if which_key == "H5File":
            for file_ in file_list:
                h5file_ = file_.split(".h5")[0]
                output_list.append(h5file_)
                
        elif which_key == "Datetime":
            for file_ in file_list:  
                dtfile_ = file_.split(".h5, ")[1].split("\n")[0]
                output_list.append(dtfile_)
 
        return output_list
    
    def making_train_valid_test_list(self): 
        self.custom_print("============== BEGIN ==================")   

        if self.is_h5seq:
            h5_path   = os.path.join(self.root, self.h5_mainfolder, "h5files") 
            all_h5_file_list = glob.glob("%s/*.h5" % h5_path)
            self.dict_["h5_path"] = h5_path  
        else: 
            h5_path = os.path.join(self.root, self.h5_mainfolder ) 
            self.dict_["pickle_data"] = os.path.join(h5_path, self.args.pickle_data ) 
            self.dict_["csv_data"] = os.path.join(h5_path, self.args.csv_data ) 
            csv_data         = pd.read_csv(self.dict_["csv_data"])
            all_h5_file_list = pd.to_datetime(csv_data["Datetime"])

        text_prefix = { "Train": "Train", "Test": "Test", "Valid": "Valid" }

        self.dict_["train_list_filename"] = "%s_%s.txt" % (text_prefix["Train"], self.setting)
        self.dict_["test_list_filename"]  = "%s_%s.txt" % (text_prefix["Test"],  self.setting)
        self.dict_["valid_list_filename"] = "%s_%s.txt" % (text_prefix["Valid"], self.setting)


        self.custom_print("Train List: %s_%s.txt" % (text_prefix["Train"], self.setting))
        self.custom_print("Test List:  %s_%s.txt" % (text_prefix["Test"],  self.setting))
        self.custom_print("Valid List: %s_%s.txt" % (text_prefix["Valid"], self.setting))

        if ((os.path.isfile(self.dict_["train_list_filename"])) or (os.path.isfile(self.dict_["test_list_filename"]))) or (os.path.isfile(self.dict_["valid_list_filename"])) :
            
            self.dict_["files_status"] = "train/test/valid already exist"
            save_settings_dict(self.h5_mainfolder, self.dict_, file_name="spliting_setting.csv") 
            self.custom_print(self.dict_["files_status"])

            list_training   = read_text2list("%s_%s.txt" % (text_prefix["Train"], self.setting))
            list_testing    = read_text2list("%s_%s.txt" % (text_prefix["Test"],  self.setting))
            list_validating = read_text2list("%s_%s.txt" % (text_prefix["Valid"], self.setting))

            self.custom_print("============== FINISH ==================")

            return list_training, list_validating, list_testing
        
        else:
            if len(all_h5_file_list) == 0: 
                self.dict_["files_status"] = "h5 does not exist"
                self.custom_print(self.dict_["files_status"])
                save_settings_dict(self.h5_mainfolder, self.dict_, file_name="spliting_setting.csv") 

                raise  ValueError('You have to find the way to list all the files because Text Files `test_/valid_/traing_....txt` does not exist. Or else you may need to generate h5files_Frame-4-Mins_IMS-/ folder by running Step 0 and Step 1')

        now = datetime.now()
        self.dict_["files_status"] = "train/test/valid are generated on %s" % now.strftime("%Y-%m-%d")  
        self.custom_print(self.dict_["files_status"])
        save_settings_dict(self.h5_mainfolder, self.dict_, file_name="spliting_setting.csv") 
 
        if self.is_h5seq:
            month_list = ["%02d"   % ind   for ind   in range(self.args.start_month, self.args.end_month + 1)]
            file_name_month_list = ["%s/%s-*-2023-**.h5" % ( h5_path, month) for month in month_list]  
        else:
            month_list = [ ind   for ind   in range(self.args.start_month, self.args.end_month + 1)] 
            csv_data_all         = pd.read_csv(self.dict_["csv_data"])
            csv_data             = csv_data_all[['Datetime', 'Month']]  
            csv_data.loc[:, "Datetime"] = pd.to_datetime(csv_data["Datetime"]) 
  
        dict_filelist_testing_month  = {}
        dict_filelist_valid_month    = {}
        dict_filelist_training_month = {}
        
        if (self.args.use_extra is not None):

            print("Splitting [train/test/valid] by using external lists")
            print("Testing    list: [%s]" % self.args.use_extra_testing)
            print("Validating list: [%s]" % self.args.use_extra_validation)
            textfile  = self.args.use_extra_testing
            validfile = self.args.use_extra_validation
 
            
            if self.is_h5seq: 
                list_testing    = self.extract_file2list(textfile, which_key="H5File")
                list_validating = self.extract_file2list(validfile, which_key="H5File")

            else: 
                list_testing    = self.extract_file2list(textfile, which_key="Datetime")
                list_validating = self.extract_file2list(validfile, which_key="Datetime")

                all_h5_file_list = pd.to_datetime(all_h5_file_list.tolist()).strftime('%Y-%m-%d %H:%M:%S').tolist()
 
            remaining_list = np.setdiff1d(all_h5_file_list, list_testing) 
            list_training  = np.setdiff1d(remaining_list, list_validating)
            list_training  = list(list_training) 

            with open(os.path.join(self.h5_mainfolder, "Test_%s.txt" % self.setting)  , 'w') as output:
                for row in list_testing:  
                    output.write(str(row) + '\n')

            with open(os.path.join(self.h5_mainfolder, "Valid_%s.txt" % self.setting)  , 'w') as output:
                for row in list_validating: 
                    output.write(str(row) + '\n')

            with open(os.path.join(self.h5_mainfolder, "Train_%s.txt" % self.setting), 'w') as output:
                for row in list_training: 
                    output.write(str(row) + '\n')     
            
        else: 
        
            if self.is_h5seq:

                for file_ in file_name_month_list:
                    file_list_per_month = glob.glob(file_)
                    self.custom_print("%s = %d" % (file_, len(file_list_per_month)))   

                for month_, file_name_each_month in zip(month_list, file_name_month_list): 

                    file_list_per_month = glob.glob(file_name_each_month)
                    file_list_per_month.sort()
                    
                    # days_list_per_month, days_dict_filelist_per_month = extract_dict_dayslist(file_list_per_month) 
                    days_list_per_month,   date2files,   file2datetime  = extract_dict_dayslist(file_list_per_month)

                    num_file_per_month  =  len(days_list_per_month)


                    if num_file_per_month > 10: 

                        # Separate file into index list  
                        testing_filelist, valid_filelist, training_filelist, file_list_per_month_org = self.separate_index_for_testing_valid_traning(month_, days_list_per_month, date2files, file2datetime, num_file_per_month)

                        dict_filelist_testing_month[month_]  = testing_filelist
                        dict_filelist_valid_month[month_]    = valid_filelist
                        dict_filelist_training_month[month_] = training_filelist 


                        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_valid_month[month_] ) == 0
                        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_training_month[month_] ) == 0
                        assert intersection(dict_filelist_valid_month[month_],    dict_filelist_training_month[month_]) == 0  
                        # assert intersection(setunion(dict_filelist_testing_month[month_],  dict_filelist_valid_month[month_],  dict_filelist_training_month[month_]), file_list_per_month_org) == len(file_list_per_month_org) 

                        self.custom_print("[%s] Testing %d / Valid %d / Training %d" % (month_, len(dict_filelist_testing_month[month_]),len(dict_filelist_valid_month[month_]) ,len(dict_filelist_training_month[month_])))

            else:
                for month_ in month_list: 

                    csv_data_month_ = csv_data.loc[(csv_data['Month'] == month_)]

                    csv_data_month_     = csv_data_month_.sort_values(by=['Datetime']) 
                    days_list_per_month = csv_data_month_['Datetime'].tolist() 
                    # days_list_per_month, date2files, file2datetime = extract_dict_dayslist(file_list_per_month) 
                    num_file_per_month  =  len(days_list_per_month) 

                    if num_file_per_month > 10: 

                        # Separate file into index list  
                        testing_filelist, valid_filelist, training_filelist, file_list_per_month_org = self.separate_index_for_testing_valid_traning(month_, days_list_per_month, num_file_per_month=num_file_per_month)

                        dict_filelist_testing_month[month_]  = testing_filelist
                        dict_filelist_valid_month[month_]    = valid_filelist
                        dict_filelist_training_month[month_] = training_filelist 
 
                        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_valid_month[month_] ) == 0
                        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_training_month[month_] ) == 0
                        assert intersection(dict_filelist_valid_month[month_],    dict_filelist_training_month[month_]) == 0  
                        # assert intersection(setunion(dict_filelist_testing_month[month_],  dict_filelist_valid_month[month_],  dict_filelist_training_month[month_]), file_list_per_month_org) == len(file_list_per_month_org) 

                        self.custom_print("[%s] Testing %d / Valid %d / Training %d" % (month_, len(dict_filelist_testing_month[month_]),len(dict_filelist_valid_month[month_]) ,len(dict_filelist_training_month[month_])))


            list_training, list_validating, list_testing = self.dict2text(dict_filelist_testing_month, dict_filelist_valid_month, dict_filelist_training_month)

        self.custom_print("============== FINISH ==================")
  

if __name__ == "__main__":

    config_file      = "h5files-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-18/making-h5-setting.csv" 
    h5_datasplitting = Spliting_H5files(configs=config_file)
    h5_datasplitting.making_train_valid_test_list()