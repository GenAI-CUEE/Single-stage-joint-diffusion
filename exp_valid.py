import torch.nn.functional as F 
from torch.utils.data import DataLoader  
import torch

import numpy as np  
import math
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import itertools
from tqdm import tqdm
import pdb
import time
import csv

from utils.config import MyTracker, Log_Object


def infer(network, dataset, config, epoch,  record_dict={}):

    dev    = "cuda:" + str(config["Training"]["gpu_id"])
    device = torch.device(dev if torch.cuda.is_available() else "cpu")


    test_image_path = os.path.join(config["logging_version_path"], "test-images")
    tracker = MyTracker(config["stats_path"], stats_image_path=test_image_path)
    print("STATS PATH: %s" % config["stats_path"])

    valid_output = network.evaluation(dataset, config, epoch, mode="test")

    valid_images_list = valid_output["samples_list"]

    valid_log = {**{k: v for k, v in valid_output.items() if k != "samples_list"}, "EP": epoch}

    tracker.test_stats.log(valid_log)
    tracker.test_stats.log_to_csv(epoch)

    for index_list_, valid_images_list_ in enumerate(valid_images_list):
        tracker.image_list(valid_images_list[index_list_]["ground_truth"], epoch = epoch, image_key= "%d-GT"   % index_list_)
        tracker.image_list(valid_images_list[index_list_]["predictions"],  epoch = epoch, image_key= "%d-Pred" % index_list_)

    # SkyGPT-only: genuine forecast performance (predicts the truly-future
    # Y_image via forecast()'s chained sample() rollout), separate from the
    # run_model()-based reconstruction metrics logged above. Guarded on
    # which_model since forecast() has no VQVAE analogue (asserts against it).
    if hasattr(network, "forecast") and getattr(network, "which_model", None) != "VQVAE":
        forecast_output = network.forecast(dataset, config, epoch, mode="test")

        forecast_images_list = forecast_output["samples_list"]

        forecast_log = {**{k: v for k, v in forecast_output.items() if k != "samples_list"}, "EP": epoch}

        forecast_stats = Log_Object(config["stats_path"], key="Forecast")
        forecast_stats.log(forecast_log)
        forecast_stats.log_to_csv(epoch)

        for index_list_, _ in enumerate(forecast_images_list):
            tracker.image_list(forecast_images_list[index_list_]["ground_truth"], epoch = epoch, image_key= "Forecast-%d-GT"   % index_list_)
            tracker.image_list(forecast_images_list[index_list_]["predictions"],  epoch = epoch, image_key= "Forecast-%d-Pred" % index_list_)
