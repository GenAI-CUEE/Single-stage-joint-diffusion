import zipfile
import os
from pathlib import Path
import pdb
import glob
import csv
import sys
from datetime import datetime
import pandas as pd

import pytz

def remove_comments(text_file):
    with open(text_file, "r", errors='ignore') as file:
        text = file.read()
 
    lines     = text.strip().split('\n')	# splitting lines  
    org_lines = lines.copy() 

    comments = []	# to store commented lines for removal
    for line in lines:
        if(line[0] == "#"):
            comments.append(line)

    for line in comments:
        lines.remove(line)
     
    return lines, org_lines



def get_elements(result_lines):
    data_list = []
    for aline in result_lines:
        aline_ =  aline.split(" ")
        aline_list = []
        for ele_ in aline_:
            if not(ele_ == ""): 
                aline_list.append(ele_)

        if (len(aline_list)) == 33:
            data_list.append(aline_list)  
    return data_list



def get_datetime_and_ghi(data_list, extract_data_type=["I"]):  
    data_type_2_column = {"I": 3, "Diff": 2,  "C": 18, "P": 21, "H": 22}
    list_of_dict = []
    for data_line in data_list:  
        timestamp = pd.Timestamp(data_line[0], tz=None)   # pd.Timestamp(data_line[0], tz='Europe/Paris') 
        local_dict = {"Datetime":timestamp} 
        for data_ in extract_data_type:
            local_dict[data_] = float(data_line[data_type_2_column[data_]])  
        list_of_dict.append(local_dict) 
    return list_of_dict



def extract_zip_files(org_folder='SIRTA_irradiance', extracted_folder='SIRTA_irradiance_solys2', year='2020', extract_data_type=["I"]): 
    
    month_list = os.listdir(os.path.join(org_folder, year))
    month_list.sort() 

    unzip_year = os.path.join(extracted_folder, year)
    os.makedirs(unzip_year, exist_ok=True)
    
    failed_file_list = []

    list_of_dict = []
    for month_ in month_list:
        day_list = os.listdir(os.path.join(org_folder, year, month_))
        day_list.sort() 
        
        extracted_month_ = os.path.join(extracted_folder, year, month_)
        os.makedirs(extracted_month_, exist_ok=True)

        for day_ in day_list:  
            path_ = glob.glob(os.path.join(org_folder, year, month_, day_, 'radflux_1b_Lz2M1minIsolys2PrayDp-QC_v01*.txt'))    
            zf    = os.path.basename(path_[-1]) #save the zipfile path

            result_lines, org_lines = remove_comments(path_[-1])  
            data_list               = get_elements(result_lines) 

            if not(len(result_lines) == len(data_list)):  
                print("not(len(result_lines) == len(data_list))") 
            
            list_of_dict_ = get_datetime_and_ghi(data_list, extract_data_type)   

            list_of_dict += list_of_dict_ 

            with open('%s/failed_to_extract.txt' % extracted_month_, 'w') as f:
                for line in failed_file_list:
                    f.write(f"{line}\n")

    return list_of_dict    
                


if __name__ == "__main__":
    # Usage: python Step0_1_Extract_irradiance.py [year]
    org_folder       = 'SIRTA_irradiance'
    extracted_folder = 'SIRTA_irradiance_solys2_ICPH-daytime'
    os.makedirs(extracted_folder, exist_ok=True)

    data_type_ = ["I", "Diff", "C", "P", "H"]

    year_list = [sys.argv[1] if len(sys.argv) > 1 else "2023"]
    for year_ in year_list:
        list_of_dict = extract_zip_files(org_folder=org_folder, extracted_folder=extracted_folder, year=year_, extract_data_type=data_type_) 
        df = pd.DataFrame(list_of_dict)
        df.to_csv(os.path.join(extracted_folder, "%s.csv" % year_), index=False)
         