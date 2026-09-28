import zipfile
import os
from pathlib import Path
import pdb
import glob
import csv
import sys
from datetime import datetime
import pandas as pd
import numpy as np
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



def get_datetime_and_ghi(data_list): 
    list_of_dict = []
    for data_line in data_list:  
        timestamp = pd.Timestamp(data_line[0], tz=None)  
        list_of_dict.append({"Datetime":timestamp, "I": float(data_line[3])}) 
    return list_of_dict



def clean_irradiance(csv_path= 'SIRTA_irradiance_solys2/2023.csv'): 
    
    df_csv = pd.read_csv(csv_path)   

    df_csv.loc[df_csv["I"] < 0.0, "I"] = 0
    df_csv = df_csv.replace('', np.nan).dropna()  

    return df_csv    
                


if __name__ == "__main__":
    # Usage: python Step0_2_Clean_irradiance.py [year]
    extracted_folder = 'SIRTA_irradiance_solys2_ICPH-daytime'
    os.makedirs(extracted_folder, exist_ok=True)

    year_list = [sys.argv[1] if len(sys.argv) > 1 else "2023"]
    for year_ in year_list:
        csv_input_path = os.path.join(extracted_folder, "%s.csv" % year_)
        df_csv = clean_irradiance(csv_path=csv_input_path) 

        csv_output_path = os.path.join(extracted_folder, "%s-cleaned.csv" % year_)
        df_csv.to_csv(csv_output_path, index=False)
         