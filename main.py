import torch.utils.data
import torch.distributed as dist
import torch
import random 
import numpy as np 
import glob
import csv
import os
import pdb
 
import datetime
import yaml
import pdb
from termcolor import colored
from tqdm import tqdm
from train import train
from exp_valid import infer
from dataloader.sirta_dataset import SeqDataset_SIRTA

from models import SkyGPT, PhyDNet, PhyDNetGAN, ConvLSTM, Xiong, DLinear, PatchTST, TimeXer, Sunset, Text2ImageJoint
from utils.config import custom_print, recursive_print_dict, set_folder_name, apply_version_note


class process:
    def __init__(self, config_file = None ):  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
             
        self.config       = config
        self.data_mode    = config["Data_mode"]
        self.network_name = config["Network_name"]

        if config["Training"]["random_seed"]: 
            fix_seed     = config["Training"]["random_seed"] 
            random.seed(fix_seed)
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)

        # Setting folder name in here : set_folder_name()
        self.mainpath = set_folder_name(config)  

        if config["Network_name"] == "Skygpt":
            self.which_model = "VQVAE" if config["Training"]["train_only_VQVAE"] else "Skygpt"
            if not(config[self.which_model]["load_ckpt"]): 
                 
                # load_ckpt is False. 
                # Regardless of whether it is a VQVAE or SkyGPT, this means that you train a new model. 
                # therefore, we assign a new version.
                self.version = datetime.datetime.today().strftime("D%dM%mY%y_H%HM%M")
                self.version = apply_version_note(self.version, config[self.which_model].get("Note"))
                self.config[self.which_model]["version"] = self.version
                print(colored("Train new model", "light_magenta"))

            else:

                # if loading checkpoints, we will find the version specified in the config file... 
                if config[self.which_model]["version"]: 
                    self.version = config[self.which_model]["version"] # get_version_from_ckpt(config, self.data_mode, self.network_name)
                    print(colored("Load trained model version: %s" % config[self.which_model]["version"], "cyan")) 
                else: 
                    # We could not get the existing version.
                    # This means that you will train a new model.   
                    print(colored("We could not get the existing version. \n " \
                    "Specify the version (e.g. version: D09M02Y26_H16M11) or set [load_ckpt: False]", "light_magenta")) 
                    KeyError

        else:
        
            if not(config["load_ckpt"]): 
                # load_ckpt is False. 
                # This means that you train a new model.
                # therefore, we assign a new version.
                self.version = datetime.datetime.today().strftime("D%dM%mY%y_H%HM%M")
                self.version = apply_version_note(self.version, config.get("Note"))
                self.config["version"] = self.version
                print(colored("Train new model", "light_magenta"))

            else:
                # if loading checkpoints, we will find the version specified in the config file... 
                if config["version"]: 
                    self.version = config["version"] # get_version_from_ckpt(config, self.data_mode, self.network_name)
                    print(colored("Load trained model version: %s" % config["version"], "cyan")) 
                else:
                    # We could not get the existing version.
                    # This means that you will train a new model.
                    self.version = datetime.datetime.today().strftime("D%dM%mY%y_H%HM%M")
                    self.version = apply_version_note(self.version, config.get("Note"))
                    self.config["version"] = self.version
                print(colored("Train new model", "light_magenta"))
    

        self.config["logging_path"] = os.path.join(config["Main_logging_path"], "%s" % self.mainpath) 
        self.config["logging_version_path"] = os.path.join(config["Main_logging_path"], "%s" % self.mainpath, "%s" % self.version )   
        
        self.config["stats_path"] = os.path.join(config["Main_logging_path"], "%s" % self.mainpath, "%s" % self.version, "stats")
        self.config["ckpt_path"]  = os.path.join(config["Main_logging_path"], "%s" % self.mainpath, "%s" % self.version, "checkpoints") 
        print("The processing will operates on the following paths:") 
        print("Main_logging_path : %s" % config["Main_logging_path"])
        print(">>>> logging_path : %s" %  config["logging_path"])
        print(">>>> ** VERSION : [%s] ** <<<<" % self.version)
        print(">>>> logging_version_path : %s" % config["logging_version_path"])
        print(">>>> stats_path   : %s" % config["stats_path"]) 
        print(">>>>  ckpt_path   : %s" % config["ckpt_path"])  


        os.makedirs(config["Main_logging_path"], exist_ok=True)
        os.makedirs(config["logging_path"], exist_ok=True)
        os.makedirs(config["logging_version_path"], exist_ok=True)
        os.makedirs(config["stats_path"],   exist_ok=True)
        os.makedirs(config["ckpt_path"],    exist_ok=True)

        self.stats_path = config["stats_path"]
        self.ckpt_path  = config["ckpt_path"]  

        self.setting_path = os.path.join(self.stats_path, "settings-%s.txt" % self.version) 
        self.setting_yaml = os.path.join(self.stats_path, "settings-%s.yaml" % self.version)  
        
        self.config["setting_path"] = self.setting_path
        self.config["setting_yaml"] = self.setting_yaml

        #self._save_settings_dict(self.config)  
        

    def _save_settings_dict(self, config, verbose=True): 
        custom_print("================ save settings ================ ")
        recursive_print_dict(config, logfile_path = self.setting_path)         
        custom_print("=============================================== \n") 


    def _custom_print(self, messages):
        custom_print(messages)   
  

    def _get_previous_training(self, module):

        load_option = "others"
         
        if (self.config["Network_name"] == "Skygpt") and (self.config["Training"]["train_only_VQVAE"] == False): 
            # Load for training the Transformer in SkyGPT
            if (self.config["Skygpt"]["load_ckpt"] == False): 
                ckpt_path    = os.path.join(self.config["Main_logging_path"], self.config["Skygpt"]["Network"]["vaque_ckp_path"]) 
                weights_list = glob.glob('%s/*.pth' % ckpt_path) 
                load_option = "skygpt-vae_checkpoint"
            else:
                ckpt_path    = self.config["ckpt_path"]
                weights_list = glob.glob('%s/checkpoint-ep*.pth' % ckpt_path) 
                
                weights_list_dict = {}
                for weight_ in weights_list:
                    ep = int(weight_.split(os.path.join(ckpt_path, "checkpoint-ep"))[-1].split(".pth")[0])
                    weights_list_dict[ep] = weight_

                if len(weights_list_dict) > 0:
                    weights_list_dict = dict(sorted(weights_list_dict.items()))
                    weights_list = list(weights_list_dict.values())
                else:
                    weights_list = glob.glob('%s/*.pth' % ckpt_path)
                weights_list.sort() 

                load_option = "skygpt-transformer_checkpoint"

        elif  (self.config["Network_name"] == "Skygpt") and (self.config["Training"]["train_only_VQVAE"] == True): 
            # Load for training the VQVAE in SkyGPT
            ckpt_path    = self.config["ckpt_path"]  
            weights_list = glob.glob('%s/*.pth' % ckpt_path) 
            weights_list.sort() 
            load_option = "skygpt-vae_checkpoint"

        elif self.config["Network_name"] == "Xiong":
            # Same generic checkpoint format as Skynet/PhyDNet/ConvLSTM
            # (checkpoint['state_dict'] -> module.model), but sorted NUMERICALLY:
            # a plain lexicographic sort picks ep99 over ep100 whenever the epoch
            # in the filename is not zero-padded.
            ckpt_path    = self.config["ckpt_path"]
            weights_list = glob.glob('%s/*.pth' % ckpt_path)

            def _ep_of(path_):
                digits = ''.join(c if c.isdigit() else ' '
                                 for c in os.path.basename(path_)).split()
                return int(digits[-1]) if digits else -1

            weights_list.sort(key=_ep_of)
            load_option = "xiong_checkpoint"

        else: 
            ckpt_path    = self.config["ckpt_path"] 
            weights_list = glob.glob('%s/*.pth' % ckpt_path)
            weights_list.sort() 

        

        if len(weights_list)> 0:   
            generator_name =  weights_list[-1] 
            self._custom_print(colored("Load state dict: [%s]" % generator_name, "cyan")) 
             
            checkpoint = torch.load(generator_name, weights_only=False)
             
            if load_option == "skygpt-vae_checkpoint" :
                module.model.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])   

            elif load_option == "skygpt-transformer_checkpoint":    
                module.model.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])  
                module.model.transformer.load_state_dict(checkpoint['transformer_state_dict'])   
                 
            else:
                module.model.load_state_dict(checkpoint['state_dict'])
            
            if hasattr(module, 'discriminator'):
                module.discriminator.load_state_dict(torch.load(generator_name, weights_only=False)['disc_state_dict'])

            if self.config["Training"]["is_train"]:
                if (load_option == "skygpt-vae_checkpoint") and (self.config["Training"]["train_only_VQVAE"] == True):
                    if self.config["Training"]["resume_optimizer"]:   
                        module.optimizer.load_state_dict(torch.load(generator_name, weights_only=False)['optimizer'])
                        print(colored("Loaded Learning rate = [%f]" % module.optimizer.param_groups[-1]['lr'], "cyan"))
                    else:
                        print(colored("<<NO OPTIMIZER LOAD>> ", "green"))

                elif (load_option == "skygpt-vae_checkpoint") and (self.config["Training"]["train_only_VQVAE"] == False):
                    print(colored("<<NO OPTIMIZER LOAD>> ", "green"))

                elif (load_option == "skygpt-transformer_checkpoint") and (self.config["Training"]["train_only_VQVAE"] == False):
                    
                    if self.config["Training"]["resume_optimizer"]:   
                        module.optimizer.load_state_dict(torch.load(generator_name, weights_only=False)['optimizer'])
                        print(colored("Loaded Learning rate = [%f]" % module.optimizer.param_groups[-1]['lr'], "cyan"))
                    else:
                        print(colored("<<NO OPTIMIZER LOAD>> ", "green"))

                elif load_option == "xiong_checkpoint":
                    # Each Xiong stage builds its optimizer over a DIFFERENT
                    # parameter set, so a checkpoint written by the previous
                    # stage cannot be loaded into this stage's optimizer.
                    # Resuming is only valid within the same stage.
                    if self.config["Training"]["resume_optimizer"]:
                        try:
                            module.optimizer.load_state_dict(
                                torch.load(generator_name, weights_only=False)['optimizer'])
                            print(colored("Loaded Learning rate = [%f]"
                                          % module.optimizer.param_groups[-1]['lr'], "cyan"))
                        except (KeyError, ValueError) as e:
                            print(colored("<<NO OPTIMIZER LOAD>> (%s) -- expected when "
                                          "the checkpoint came from a different stage"
                                          % type(e).__name__, "yellow"))
                    else:
                        print(colored("<<NO OPTIMIZER LOAD>> ", "green"))

                else: 
                    print(colored("<<Debug VQVAE>> NO OPTIMIZER LOAD", "green"))
                
                epoch_start = torch.load(generator_name, weights_only=False)['epoch'] 
                self._custom_print(colored("[Reload epoch: %05d]" % epoch_start, "cyan"))   
                epoch_start = epoch_start + 1

            else:
                print('Infer mode: loaded %s model from %s' %  (self.config["Network_name"], generator_name))
                epoch_start = torch.load(generator_name, weights_only=False)['epoch']

        else:
            if self.config["Training"]["is_train"]:
                self._custom_print(colored("Start new training! at ep 0", "magenta"))
            else:
                self._custom_print(colored(
                    "Infer mode: no checkpoint found in %s -- evaluating the model "
                    "as constructed by Model.__init__ (e.g. any warm start it already "
                    "applied on its own, such as Text2ImageJoint's pretrained_ckpt_dir/file)"
                    % self.config["ckpt_path"], "magenta"))
            epoch_start = 0
            record_dict = {}

        return module, epoch_start # start_file_index, iter_start

    def main(self): 

        """The main function."""
        # Initialize network
        self._custom_print("Training Network: %s" % self.network_name)

        if self.network_name =='ConvLSTM':
            module = ConvLSTM.Model(self.config) 

        elif self.network_name =='PhyDNet':
            module = PhyDNet.Model(self.config)

        elif self.network_name =='PhyDNetGAN':
            module = PhyDNetGAN.Model(self.config)

        elif self.network_name =='Skygpt':
            module = SkyGPT.Model(self.config)  

        elif self.network_name =='Xiong':
            module = Xiong.Model(self.config)

        elif self.network_name =='DLinear':
            module = DLinear.Model(self.config)

        elif self.network_name =='PatchTST':
            module = PatchTST.Model(self.config)

        elif self.network_name =='TimeXer':
            module = TimeXer.Model(self.config)

        elif self.network_name =='Sunset':
            module = Sunset.Model(self.config)

        elif self.network_name =='Text2ImageJoint':
            module = Text2ImageJoint.Model(self.config)

        self.config["Training"]["Number of parameters"] = module.num_param
        self._save_settings_dict(self.config) 

        with open(self.setting_yaml, 'w') as f:
            yaml.dump(self.config, f)  

        is_train = self.config["Training"]["is_train"]
        

        if is_train:  
            loaded_module, epoch_start = self._get_previous_training(module) 
 
            if self.data_mode == "SIRTA16x15":
                self._custom_print("Training on [%s]" % self.data_mode)
                dataset = SeqDataset_SIRTA

            last_lr = train(loaded_module, dataset, self.config, self.data_mode, epoch_start)
             
            self.config["last_lr"] = last_lr
            self._save_settings_dict(self.config) 
 

        else:      
            loaded_model, loaded_epoch = self._get_previous_training(module) 
            if self.data_mode == "SIRTA16x15":
                self._custom_print("Inference on [%s]" % self.data_mode)
                dataset = SeqDataset_SIRTA

            infer(loaded_model, dataset, self.config, loaded_epoch)


 

if __name__ == "__main__":

    # Train or run inference for one of this release's configs, e.g.:
    # file_h5 = process(config_file = "configs/convlstm.yaml")
    # file_h5 = process(config_file = "configs/phydnet.yaml")
    # file_h5 = process(config_file = "configs/phydnetgan.yaml")
    # file_h5 = process(config_file = "configs/skygpt.yaml")
    # file_h5 = process(config_file = "configs/xiong.yaml")
    # file_h5 = process(config_file = "configs/dlinear_wiclr.yaml")
    # file_h5 = process(config_file = "configs/patchtst_wiclr.yaml")
    # file_h5 = process(config_file = "configs/timexer_wiclr.yaml")
    # file_h5 = process(config_file = "configs/sunset_wiclr.yaml")
    # file_h5 = process(config_file = "configs/joint_clean_seed1.yaml")
    file_h5 = process(config_file = "configs/joint_x0hat_seed1.yaml")
    file_h5.main()