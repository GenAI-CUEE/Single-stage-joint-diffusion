import argparse
import pdb
import yaml 
import os
import csv 
import torch
import numpy as np
import pandas as pd
from PIL import Image
import platform
from pathlib import Path

def convert2str(message_to_print):
    if type(message_to_print) is tuple:
        message_to_print  = "( " + ', '.join(map(str, message_to_print)) + " )"

    elif type(message_to_print) is list:
        message_to_print  = "[ " + ', '.join(map(str, message_to_print)) + " ]"

    elif isinstance(message_to_print, (int, float)):
        message_to_print = str(message_to_print)  
    return message_to_print

def custom_print(message_to_print, logfile_path: str = None): 
    print(message_to_print)
    if logfile_path is not None: 
        with open(logfile_path, 'a') as of:
            of.write(message_to_print + '\n')


def recursive_print_dict(dict_config, layer=0, logfile_path= ""):
    layer = 0
    for key, value in dict_config.items():
        if type(value) is dict:
            custom_print("%s: ..." % (key), logfile_path)
            for subkey, subvalue in value.items():
                layer_str = ">" * 4
                custom_print(layer_str + " %s:   %s" % (subkey, convert2str(subvalue)), logfile_path)
        else:
            custom_print("\n%s:   %s" % (key, convert2str(value)), logfile_path)
 

def get_version_from_ckpt(config: dict, mode: str, network: str):

    loging_path     = config[mode][network]["Paths"]["loging_path"]
    checkpoint_path = config[mode][network]["Paths"]["ckpt_path"]
    version         = checkpoint_path.split(loging_path)[1].split("checkpoint")[0] 
    
    return version


def set_folder_name(config: dict, mode: str = "SIRTA16x15"):

    dl_folder =  setting_subfolder_name(config)

    return dl_folder


def apply_version_note(version: str, note) -> str:
    """Append an optional, free-text Note to an auto-generated version
    timestamp (e.g. "D09M08Y26_H20M03" + Note "Step50" -> "D09M08Y26_H20M03_Step50"),
    matching this repo's existing manual "_Seed1"/"_Seed2" version-suffix
    convention. No-op if note is None/empty."""
    if note:
        return "%s_%s" % (version, note)
    return version


def setting_subfolder_name(args_dict):

    if args_dict["Network_name"] == "Skynet": 
        if args_dict["Learning"]["optflow_loss_enable"]:
            batas = "Int%.1E-Gd%.1E-Opt%.1E" % ( args_dict["Learning"]["beta_int"], 
                                                 args_dict["Learning"]["beta_gd"], 
                                                 args_dict["Learning"]["beta_op"])
        else: 
            batas = "Int%.1E-Gd%.1E" % ( args_dict["Learning"]["beta_int"], 
                                         args_dict["Learning"]["beta_gd"])  

        dl_folder = "%s_B%d_lr%.1E_LOSS-%s" % ( args_dict["Network_name"],  
                                               args_dict["Training"]["batch_size"], 
                                               args_dict["Learning"]["learning_rate"], 
                                               batas)


    elif args_dict["Network_name"] == "ConvLSTM":  
        batas = "Int1p0E+00"

        dl_folder = "%s_B%d_lr%.1E_LOSS-%s" % ( args_dict["Network_name"], 
                                                args_dict["Training"]["batch_size"], 
                                                args_dict["Learning"]["learning_rate"], 
                                                batas)


    elif args_dict["Network_name"] == "PhyDNet":
        # Plain, non-adversarial PhyDNet (vincent-leguen/PhyDNet) -- a single
        # generator optimizer, no discriminator, so the folder name only
        # encodes the generative/regularization loss weights and one learning
        # rate (unlike PhyDNetGAN below, which also has a discriminator).
        batas = "Gen%.1E-Rg%.1E" % (args_dict["Learning"]["beta_generative"],
                                     args_dict["Learning"]["beta_regularization"])

        dl_folder = "%s_B%d_lr%.1E_LOSS-%s" % (args_dict["Network_name"],
                                                 args_dict["Training"]["batch_size"],
                                                 args_dict["Learning"]["learning_rate"],
                                                 batas)

    elif args_dict["Network_name"] == "PhyDNetGAN":
        # PhyDNetGAN adds a discriminator on top of PhyDNet's generator (same
        # EncoderRNN/PhyCell/ConvLSTM building blocks), so its Learning.* key
        # shape is wider (beta_discrim, epoch_enable_disc, split
        # learning_rate_gen/learning_rate_disc) -- kept as a separate
        # Network_name/results folder rather than a config toggle.
        batas = "Gen%.1E-Rg%.1E-Dsc%.1E-EPDisc%d" % (   args_dict["Learning"]["beta_generative"],
                                                        args_dict["Learning"]["beta_regularization"],
                                                        args_dict["Learning"]["beta_discrim"],
                                                        args_dict["Learning"]["epoch_enable_disc"])

        dl_folder = "%s_B%d_lrG%.1E_lrD%.1E_LOSS-%s" %( args_dict["Network_name"],
                                                        args_dict["Training"]["batch_size"],
                                                        args_dict["Learning"]["learning_rate_gen"],
                                                        args_dict["Learning"]["learning_rate_disc"],
                                                        batas)

    elif args_dict["Network_name"] == "VQVAE":
 
        batas = "Rec%.1E-Com%.1E" % (   args_dict["Learning"]["beta_reconstruction"], args_dict["Learning"]["beta_commitment"] )   
        dl_folder = "%s_B%d_lr%.1E-%s" % (args_dict["Network_name"], args_dict["Training"]["batch_size"], eval(args_dict["Learning"]["learning_rate"]), batas)  

    elif args_dict["Network_name"] == "Skygpt": 
        
        if args_dict["Training"]["train_only_VQVAE"]:
            batas = "Rec%.1E-Com%.1E" % (   args_dict["VQVAE"]["Learning"]["beta_reconstruction"], 
                                         args_dict["VQVAE"]["Learning"]["beta_commitment"] )   
            
            downsample_tuple = eval(args_dict["VQVAE"]["Network"]["downsample"])

            dl_folder = "%s-VQVAE_B%dlr%.1E_nC%d-RL%d-Em%d-Hd%d-DS-%d-%d-%d_%s" % (args_dict["Network_name"], 
                                                   args_dict["Training"]["batch_size"], 
                                                   eval(args_dict["VQVAE"]["Learning"]["learning_rate"]),   
                                                   args_dict["VQVAE"]["Network"]["n_codes"], 
                                                   args_dict["VQVAE"]["Network"]["n_res_layers"], 
                                                   args_dict["VQVAE"]["Network"]["embedding_dim"], 
                                                   args_dict["VQVAE"]["Network"]["n_hiddens"], 
                                                   downsample_tuple[0], downsample_tuple[1], downsample_tuple[2], 
                                                   batas)  

        else:
            # Will carve out only the version of the Skygpt-VQVAE's path: 
            #   args_dict["Skygpt"]["Network"]["vaque_ckp_path"] = "Skygpt-VQVAE_B16lr3p0E-04-Rec1p7E+01-Com2p5E-01/D05M08Y25_H17M05/checkpoints/checkpoint.pth"
            #   dl_folder_base = args_dict["Skygpt"]["Network"]["vaque_ckp_path"].split("/")[0]
            #   version        = args_dict["Skygpt"]["Network"]["vaque_ckp_path"].split("/")[1]
            #   >> dl_folder_base = "Skygpt-VQVAE-%s" % version
             

            if platform.system() == "Windows":
                print("Running on Windows")
                dl_folder_base_Path = Path("%s" % os.path.dirname(args_dict["Skygpt"]["Network"]["vaque_ckp_path"]))
                dl_folder_base = dl_folder_base_Path.parents[0]
            elif platform.system() == "Linux":
                print("Running on Linux")  
                dl_folder_base_Path = Path("%s" % os.path.dirname(args_dict["Skygpt"]["Network"]["vaque_ckp_path"]))
                dl_folder_base = dl_folder_base_Path.parents[0]
                # dl_folder_base = "Skygpt-VQVAE-%s" % args_dict["Skygpt"]["Network"]["vaque_ckp_path"].split("/")[1]
 
            # batas = "NStd%.1E-Krn%d-AttDrp%.1f-hDim%d-NCndFrm%d" % (args_dict["Skygpt"]["Network"]["normal_std"], 
            #                                                         args_dict["Skygpt"]["Network"]["kernel_size"], 
            #                                                         args_dict["Skygpt"]["Network"]["attn_dropout"], 
            #                                                         args_dict["Skygpt"]["Network"]["hidden_dim"], 
            #                                                         args_dict["Skygpt"]["Network"]["n_cond_frames"] )   
            
            batas = "n-cond-%d" % (args_dict["Skygpt"]["Network"]["n_cond_frames"] )  
            
            dl_folder_trns = "%s_B%dlr%.1E-%s" % (args_dict["Network_name"], 
                                                  args_dict["Training"]["batch_size"], 
                                                  eval(args_dict["Skygpt"]["Learning"]["learning_rate"]), 
                                                  batas)  
            dl_folder = "%s-TRNS-%s" % (dl_folder_trns, dl_folder_base)

    elif args_dict["Network_name"] == "Xiong":
        # Xiong et al. (2025) multimodal probabilistic forecasting.
        # Three training stages share this branch; the stage goes in the folder
        # name so they land in separate results directories.
        #
        # WHAT IS DELIBERATELY *NOT* IN THE NAME, and why:
        #   ddim_steps, ddim_eta, sde_noise_scale, n_ensemble,
        #   add_observation_noise, forecast_ddim_steps
        # These are inference-time knobs that get changed AFTER training (the
        # sde_noise_scale calibration sweep, the observation-noise ablation).
        # Anything in the folder name has to be byte-identical at train time and
        # at load time, or `load_ckpt: True` + `version:` resolves to a path that
        # does not exist. Only architecture and optimiser settings go in here.

        stage = args_dict["Training"].get("xiong_stage", "multimodal")
        learn = args_dict["Xiong"]["Learning"][stage]
        lr = learn["learning_rate"]
        lr = eval(lr) if isinstance(lr, str) else lr          # '3e-4' -> float

        V = args_dict["Xiong"]["Video"]
        R = args_dict["Xiong"]["Resnet"]
        M = args_dict["Xiong"]["Multimodal"]
        G = args_dict["Xiong"]["GP"]

        if stage == "video":
            batas = "Enc%d-Mo%d-Un%d-T%d" % (V["enc_base"], V["motion_ch"],
                                             V["unet_base"], V["num_train_timesteps"])
        elif stage == "resnet":
            batas = "B%d-Ft%d-SN%.1f-Aug%.2f-%.1f" % (
                R["base"], R["feat_dim"], R["spectral_c"],
                R["aug_noise_std"], R["aug_blur_sigma_max"])
        elif stage == "multimodal":
            batas = "Dm%d-Hd%d-El%d-Dl%d-RFF%d" % (
                M["d_model"], M["n_heads"], M["e_layers"], M["d_layers"], G["n_rff"])
        else:
            raise KeyError("Training.xiong_stage must be video|resnet|multimodal, "
                           "got %r" % stage)

        tgt = "CSI" if args_dict["Xiong"]["Target"]["mode"] == "csi" else "GHIZ"

        dl_folder = "%s-%s_B%dlr%.1E-%s-%s" % (args_dict["Network_name"],
                                               stage.upper(),
                                               args_dict["Training"]["batch_size"],
                                               lr, batas, tgt)

    elif args_dict["Network_name"] in ("DLinear", "PatchTST", "TimeXer", "Sunset"):
        # Group 3 scalar baselines (DLinear/PatchTST/TimeXer) plus Sunset
        # (image-CNN) share the same Training/<Name>.Network/<Name>.Learning
        # key shape, so one branch covers all four -- the Network_name itself
        # keeps their results directories apart.
        net = args_dict[args_dict["Network_name"]]["Network"]
        learn = args_dict[args_dict["Network_name"]]["Learning"]

        iclr_tag = "wIclr" if net.get("include_future_iclr", False) else "noIclr"

        dl_folder = "%s_B%dlr%.1E-%s" % (args_dict["Network_name"],
                                         args_dict["Training"]["batch_size"],
                                         learn["learning_rate"],
                                         iclr_tag)

    elif args_dict["Network_name"] == "Text2ImageJoint":
        # Diffusion-based joint image+GHI forecaster (see models/Text2ImageJoint.py).
        # Training uses train_batch_size/valid_batch_size/test_batch_size (not a
        # single Training.batch_size like Xiong/Group3), so that's what's read
        # here. The arm (scalar_image_source) keeps x0hat/clean/etc. runs apart.
        net = args_dict["Text2ImageJoint"]["Network"]
        learn = args_dict["Text2ImageJoint"]["Learning"]
        arm = net.get("scalar_image_source", "clean")
        # Same weights/arm as x0hat, only the eval-time conditioning differs -- named
        # as its own ORACLE folder so it can never collide with (or be mistaken for)
        # the x0hat headline run's own results directory.
        folder_arm = "ORACLE" if net.get("oracle_scalar_eval", False) else arm.upper()

        dl_folder = "%s-%s_B%dlr%.1E" % (args_dict["Network_name"],
                                         folder_arm,
                                         args_dict["Training"]["train_batch_size"],
                                         learn["learning_rate"])

    if "dl_folder" not in dir():
        # Previously this fell through and raised UnboundLocalError three lines
        # later, which points at the wrong place entirely.
        raise KeyError("setting_subfolder_name has no branch for Network_name=%r. "
                       "Add one, or the results directory cannot be named."
                       % args_dict["Network_name"])

    if args_dict["Training"]["subsampling_enable"]:
        dl_folder = dl_folder + "_TrSb%.2f" % args_dict["Training"]["subsampling_rate"]

    dl_folder = dl_folder.replace(".", "p") 
    return dl_folder
    

class Log_Object:
    def __init__(self, stats_csv_path, key: str):
        self.log_list = []  
        self.stats_csv_path = stats_csv_path
        self.key = key
        existing_csv_file = os.path.join(self.stats_csv_path, "%s.csv" % self.key)
        if os.path.isfile(existing_csv_file):
            pd_csv = pd.read_csv(existing_csv_file, index_col=0) 
            list_of_dicts = pd_csv.to_dict(orient='records')
            self.log_list = list_of_dicts

    def log(self, dict_record): 
        self.log_list.append(dict_record)

    def log_to_csv(self, epoch):
        stats_pd = pd.DataFrame(self.log_list)
        stats_pd.to_csv(os.path.join(self.stats_csv_path, "%s.csv" % self.key)) 
        


class MyTracker:
    def __init__(self, stats_csv_path, stats_image_path=None):
        
        if stats_image_path is None:
            self.stats_csv_path   = stats_csv_path
            self.stats_image_path = stats_csv_path 
            os.makedirs(self.stats_image_path, exist_ok = True)
        else:
            self.stats_csv_path   = stats_csv_path
            self.stats_image_path = stats_image_path 
            os.makedirs(self.stats_csv_path,   exist_ok = True)
            os.makedirs(self.stats_image_path, exist_ok = True)

        self.train_stats = Log_Object(stats_csv_path, key="Train")
        self.valid_stats = Log_Object(stats_csv_path, key="Valid")
        self.test_stats  = Log_Object(stats_csv_path, key="Test") 
    
    def set_stats_epoch_path(self, epoch): 
        self.stats_image_path_epoch = os.path.join(self.stats_image_path, "%05d-images" % epoch)
        os.makedirs(self.stats_image_path_epoch, exist_ok = True)

    def image(self, image:torch.Tensor, image_path:str):    
        ndarr = image.permute(1, 2, 0).to("cpu", torch.uint8).numpy() 
        im    = Image.fromarray(ndarr)
        im.save(image_path)

    def image_list(self, image_tensors:torch.Tensor, epoch:int, image_key= "IMG"): 
        self.set_stats_epoch_path(epoch)
        image_list = list(image_tensors)
        for index, img_ in enumerate(image_list):
            image_path = os.path.join(self.stats_image_path_epoch, "%s-%d.png" % (image_key, index))
            self.image(img_, image_path)




class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, model, path, epoch, val_loss, epoch_PSNR, epoch_disc_loss=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path, epoch, val_loss, epoch_PSNR, epoch_disc_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path, epoch, val_loss, epoch_PSNR, epoch_disc_loss)
            self.counter = 0

    def save_checkpoint(self, network, path, epoch, val_loss,  epoch_PSNR, epoch_disc_loss=None):

        # Dispatch on what the model actually exposes. Two changes from the
        # original, both to stop a non-SkyGPT model being misrouted:
        #
        #  (a) The SkyGPT two-part branch now requires vqvae AND transformer.
        #      It used to test `transformer` ALONE, so any model with a
        #      `.transformer` submodule and no `.vqvae` would enter it and die
        #      on `network.model.vqvae`. Behaviour for SkyGPT is unchanged --
        #      its model has both when training the transformer, and no
        #      `.transformer` attribute when training the VQVAE alone.
        #
        #  (b) The generic branch is restored. It used to print "Cannot find
        #      where to save" and then evaluate the bare name `KeyError`, which
        #      is a no-op expression, NOT a raise -- so `state` stayed undefined
        #      and torch.save() below failed with an UnboundLocalError that
        #      named the wrong culprit. Every model on main.py's generic
        #      checkpoint path (ConvLSTM, PhyDNet-without-disc, Xiong)
        #      needs this branch.

        if hasattr(network.model, 'vqvae') and hasattr(network.model, 'transformer'):
            # SkyGPT, transformer stage
            state = {
                'epoch': epoch,
                'vqvae_state_dict': network.model.vqvae.state_dict(), 
                'transformer_state_dict': network.model.transformer.state_dict(), 
                'optimizer':  network.optimizer.state_dict(), 
                'train_loss':val_loss,
                'train_disc_loss':epoch_disc_loss,
                'train_psnr':epoch_PSNR 
                }   


        elif hasattr(network.model, 'vqvae'): 
            # SkyGPT, VQVAE-only stage
            state = {
                'epoch': epoch,
                'vqvae_state_dict': network.model.vqvae.state_dict(), 
                'optimizer':  network.optimizer.state_dict(), 
                'train_loss':val_loss,
                'train_disc_loss':epoch_disc_loss,
                'train_psnr':epoch_PSNR 
                }       
            
        elif hasattr(network, 'discriminator'): 

            state = {
                'epoch': epoch,
                'state_dict': network.model.state_dict(),
                'disc_state_dict': network.discriminator.state_dict(),
                'optimizer':  network.optimizer.state_dict(),
                'disc_optimizer': network.discriminator_optimizer.state_dict(),
                'train_loss':val_loss,
                'train_disc_loss':epoch_disc_loss,
                'train_psnr':epoch_PSNR 
                }
                      
        else: 
            # Generic single-module models. For Xiong this holds the WHOLE
            # XiongNet (video + resnet + transformer + GP), which is what makes
            # the stage 1 -> 2 -> 3 warm-start chain work off one file, and what
            # main.py's "xiong_checkpoint" branch reads back.
            state = {
                'epoch': epoch,
                'state_dict': network.model.state_dict(), 
                'optimizer':  network.optimizer.state_dict(), 
                'train_loss':val_loss ,
                'train_disc_loss':epoch_disc_loss,
                'train_psnr':epoch_PSNR 
                }
            
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(state, path + '/' + 'checkpoint-ep%04d.pth' % epoch)
        self.val_loss_min = val_loss
