import zipfile
import os
from pathlib import Path
import pdb
import glob
import sys


def extract_zip_files(zip_folder='SIRTA_skyimages', unzip_folder='SIRTA_skyimages_unzip', year='2020'): 
    
    month_list = os.listdir(os.path.join(zip_folder, year))
    month_list.sort() 

    unzip_year = os.path.join(unzip_folder, year)
    os.makedirs(unzip_year, exist_ok=True)
    
    failed_file_list = []


    for month_ in month_list:
        day_list = os.listdir(os.path.join(zip_folder, year, month_))
        day_list.sort() 
        
        unzip_month_ = os.path.join(unzip_folder, year, month_)
        os.makedirs(unzip_month_, exist_ok=True)

        for day_ in day_list:  
            path_ = glob.glob(os.path.join(zip_folder, year, month_, day_, 'srf02*.zip'))    
            zf    = os.path.basename(path_[-1]) #save the zipfile path

            try: 
                print("Extracting contents from %s" % zf)
                zip_ref = zipfile.ZipFile(path_[-1], 'r') 
                zip_ref.extractall(unzip_month_) 
 
            except: 
                print("Fail to extract from %s" % zf)

                failed_file_list.append(zf)
             
            with open('%s/failed_to_extract.txt' % unzip_folder, 'w') as f:
                for line in failed_file_list:
                    f.write(f"{line}\n")
        
                


if __name__ == "__main__":
    # Usage: python Step0_0_Extract_downloaded_image.py [year]
    os.makedirs("SIRTA_skyimages_unzip", exist_ok=True)

    year_list = [sys.argv[1] if len(sys.argv) > 1 else "2023"]
    for year_ in year_list:
        extract_zip_files(zip_folder='SIRTA_skyimages', unzip_folder='SIRTA_skyimages_unzip', year=year_)