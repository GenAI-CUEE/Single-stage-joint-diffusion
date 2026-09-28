import numpy as np
import math
import random


def reserve_schedule_sampling_exp(epoch, log_length, sampling_step_1 = 15, sampling_step_2 = 30, r_exp_alpha = 2.5):

    real_input_flag_encoder = np.zeros(log_length, dtype=bool)
    
    if epoch < sampling_step_1:
        r_eta = 0.5
    elif epoch < sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(epoch - sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    for i in range(log_length):
        real_input_flag_encoder[i] = True if random.random() < r_eta else False
    
    return r_eta, real_input_flag_encoder

def schedule_sampling(epoch,pred_length, sampling_step_1 = 15, sampling_step_2 = 30):
    
    real_input_flag_decoder = np.zeros(pred_length, dtype=bool)
    
    if epoch < sampling_step_1:
        eta = 0.5
    elif epoch < sampling_step_2:
        eta = 0.5 - (0.5 / (sampling_step_2 - sampling_step_1)) * (epoch - sampling_step_1)
    else:
        eta = 0
    
    for i in range(pred_length):
        real_input_flag_decoder[i] = True if random.random() < eta else False
    
    return eta, real_input_flag_decoder