# Adapted from SkyGPT (github.com/yuhao-nie/SkyGPT, MIT), which builds on VideoGPT (github.com/wilson1yan/VideoGPT, MIT). See NOTICE.
import os
import math
import time
import pdb 
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler 

from .vqvae import VQVAE           # VQ-VAE from VideoGPT via SkyGPT (MIT); see NOTICE

from .resnet import resnet34
from .utils import shift_dim, charbonnier_Loss
from .constrain_moments import K2M 
from .transformer import Transformer
 

class Skygpt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        # Path to trained VQ-VAE model
        if args["Training"]["train_only_VQVAE"]:
            # if ver 1 :
            #     from .vqvae_ver1 import VQVAE
            # elif ver 2:
            #     from .vqvae_ver2 import VQVAE

            self.vqvae = VQVAE(args["VQVAE"])
        else:
            # if ver 1 :
            #     from .vqvae_ver1 import VQVAE
            # elif ver 2:
            #     from .vqvae_ver2 import VQVAE

            self.vqvae = VQVAE(args["VQVAE"])

            for p in self.vqvae.parameters():
                p.requires_grad = False
            self.vqvae.codebook._need_init = False
            self.vqvae.eval()

        # Non-VQVAE part of the model: the physics-informed transformer prior
        self.transformer = Transformer(
            args,
            latent_shape=self.vqvae.latent_shape,
            embedding_dim=self.vqvae.embedding_dim,
            n_codes=self.vqvae.n_codes,
        )

    # def set_training(self, is_training):
    #     self.transformer.set_training(is_training)

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, video, label=None):
        device = self.transformer.fc_in.weight.device

        self.transformer.set_training(False)

        video = video.permute(0, 2, 1, 3, 4)
        cond = dict()
        if self.transformer.use_frame_cond or self.transformer.class_cond:
            assert video is not None
            video = video # batch['video']
            if self.transformer.class_cond:
                label = label # batch['label']
                cond['class_cond'] = F.one_hot(label, self.args["Skygpt"]["Network"]["class_cond_dim"]).type_as(video)
            if self.transformer.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.args["Skygpt"]["Network"]["n_cond_frames"]]

        samples = torch.zeros((n,) + self.transformer.shape).long().to(device)


        idxs = list(itertools.product(*[range(s) for s in self.transformer.shape]))
        sample_time = int(time.time())

        with torch.no_grad():
            prev_idx = None
            max_idx = 0 
            _, base_embeddings = self.vqvae.encode(cond['frame_cond'], include_embeddings=True)
            base_embeddings = shift_dim(base_embeddings, 1, -1)
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                if idx[1] == 0 and idx[2] == 0:
                    # if idx[0] <= len(cond['frame_cond']):
                    if idx[0] == 0:
                        _, base_embeddings = self.vqvae.encode(cond['frame_cond'], include_embeddings=True)
                        base_embeddings = shift_dim(base_embeddings, 1, -1)
                    else:                      
                        base_embeddings = torch.cat([base_embeddings, self.vqvae.codebook.dictionary_lookup(samples)[:, idx[0] - 1][:, None]], axis=1)
                    reuse_phycell = False
 
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)
                
                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]
                
                # base_embeddings should have the size of 50,2,16,16,256
                # newly added
                logits = self(embeddings_slice, cond, decode_step=i, decode_idx=idx, x_full=base_embeddings)
                
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1) 
                
                prev_idx = batch_idx_slice
                reuse_phycell = True
        recon_samples = self.vqvae.decode(samples)
        return  recon_samples #, self.vqvae.decode(samples_phycell), self.vqvae.decode(samples_transformer) # BCTHW in [0, 1]

    def forward(self, x, cond, decode_step=None, decode_idx=None,  x_full=None):
        return self.transformer(x, cond, decode_step=decode_step, decode_idx=decode_idx, x_full=x_full)

 
    def run_model(self, video, vqvae_target=None, vqvae_x=None, label=None, is_training_attention = True):
        
        self.transformer.attn_stack.training = is_training_attention
        
        cond = dict()
        if self.transformer.class_cond:
            #label = batch['label']
            cond['class_cond'] = F.one_hot(label, self.args["Skygpt"]["Network"]["class_cond_dim"]).type_as(x)

        if vqvae_target is None:
            self.vqvae.eval()
            with torch.no_grad():
                # (B, P, C, H, W) -> (B, C, P, H, W); permute BEFORE slicing so
                # n_cond_frames slices the time dimension, not channels.
                
                    video      = video.permute(0, 2, 1, 3, 4)

                    if self.transformer.use_frame_cond:
                        cond['frame_cond'] = video[:, :, :self.args["Skygpt"]["Network"]["n_cond_frames"]]

                    targets, x  = self.vqvae.encode(video, include_embeddings=True)
        else:
            if self.transformer.use_frame_cond:
                video      = video.permute(0, 2, 1, 3, 4)
                cond['frame_cond'] = video[:, :, :self.args["Skygpt"]["Network"]["n_cond_frames"]]

            targets, x  = vqvae_target, vqvae_x

        # x = B, 256, 4, 16, 16
        x = x.permute(0, 2, 3, 4, 1)  #x = shift_dim(x, 1, -1)
 
        logits = self(x, cond)   

        # change shape from .permute(0, 2, 1, 3, 4) , Time   
        # change from targets to use logits, SSK
        argmax_result = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1)  
 

        predict = self.vqvae.decode(argmax_result.to(torch.int64))
        # logits is now one latent time-step shorter than targets (position 0
        # dropped inside Transformer.forward to avoid leaking each position's
        # own token into its own prediction) -- drop targets' first position
        # to match instead of cropping logits to targets' full length.
        return predict, logits, targets[:, 1:]



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')

        # SkyGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=576)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
        parser.add_argument('--attn_dropout', type=float, default=0.3)

        return parser 

 
    
 