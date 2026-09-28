import glob
import pdb
import sys
import numpy as np
from sirta_utils.train_valid_test_splitting_sequence import Spliting_H5files

if __name__ == "__main__":
    # Usage: python Step4_Spliting_image_and_local-datetime_pickles_for_testing_valid_training.py [year] [imsize]

    YEAR=int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    IMSIZE=int(sys.argv[2]) if len(sys.argv) > 2 else 64

    config_file = "sirta_%04d_image%d-localtime_ICHP-daytime-dhi/making-h5-setting.csv"  % (YEAR, IMSIZE)
    h5_datasplitting = Spliting_H5files(configs=config_file)
    h5_datasplitting.making_train_valid_test_list()