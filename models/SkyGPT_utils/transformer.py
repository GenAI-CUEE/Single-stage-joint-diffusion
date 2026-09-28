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
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .phyDNet import PhyCell

class Transformer(nn.Module):
    """Non-VQVAE part of SkyGPT: the physics-informed transformer prior
    (frame-conditioning ResNet, attention stack, PhyCell, output head)."""

    def __init__(self, args, latent_shape, embedding_dim, n_codes):
        super().__init__()

        # ResNet34 for frame conditioning
        self.class_cond     = args["Skygpt"]["Network"]["class_cond"]
        self.use_frame_cond = args["Skygpt"]["Network"]["n_cond_frames"] > 0

        if self.use_frame_cond:
            frame_cond_shape = (args["Skygpt"]["Network"]["n_cond_frames"],
                                args["VQVAE"]["Network"]["resolution"] // 4,
                                args["VQVAE"]["Network"]["resolution"] // 4,
                                args["VQVAE"]["Network"]["n_hiddens"])
            self.resnet = resnet34(1, eval(args["Skygpt"]["Network"]["resnet34_sample"]),
                                    resnet_dim=args["VQVAE"]["Network"]["n_hiddens"])
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # SkyGPT transformer
        self.shape = latent_shape
        # self.shape 4x16x16

        self.fc_in = nn.Linear(embedding_dim,
                                args["Skygpt"]["Network"]["hidden_dim"],
                                bias=False)
        self.fc_in.weight.data.normal_(std=args["Skygpt"]["Network"]["normal_std"])

        # embedding_dim = 126

        self.attn_stack = AttentionStack(self.shape,
                                        args["Skygpt"]["Network"]["hidden_dim"],
                                        args["Skygpt"]["Network"]["heads"],
                                        args["Skygpt"]["Network"]["layers"],
                                        args["Skygpt"]["Network"]["dropout"],
                                        args["Skygpt"]["Network"]["attn_type"],
                                        args["Skygpt"]["Network"]["attn_dropout"],
                                        args["Skygpt"]["Network"]["class_cond_dim"],
                                        frame_cond_shape)


        # frame_cond_shape: 16x16x16x255

        self.norm = LayerNorm(args["Skygpt"]["Network"]["hidden_dim"],
                                args["Skygpt"]["Network"]["class_cond_dim"])

        self.fc_out = nn.Linear(args["Skygpt"]["Network"]["hidden_dim"],
                                n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(n_codes,
                                                    args["Skygpt"]["Network"]["hidden_dim"]))


        # n_codes = 2048


        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.kernel_size = args["Skygpt"]["Network"]["kernel_size"]
        F_hidden_dims    = self.kernel_size ** 2

        self.softmax2d = nn.Softmax2d()

        self.phycell = PhyCell( input_dim = args["Skygpt"]["Network"]["hidden_dim"],
                                hidden_dims = F_hidden_dims,
                                kernel_size = (self.kernel_size, self.kernel_size),
                                num_layers = 1 )

        self.constraint_criterion = nn.MSELoss()
        self.constraints = torch.zeros((F_hidden_dims, self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.constraints[i * self.kernel_size + j, i, j] = 1

    def set_training(self, is_training):
        self.attn_stack.set_training(is_training)

    def forward(self, x, cond, decode_step=None, decode_idx=None,  x_full=None):

        if self.use_frame_cond:
            if decode_step is None:
                #
                cond['frame_cond'] = self.cond_pos_embd(self.resnet(cond['frame_cond']))
            elif decode_step == 0:
                #
                self.frame_cond_cache = self.cond_pos_embd(self.resnet(cond['frame_cond']))
                cond['frame_cond'] = self.frame_cond_cache
            else:

                cond['frame_cond'] = self.frame_cond_cache
        #
        h = self.fc_in(x)

        #take a long time

        try:
            h_1 = self.attn_stack(h, cond, decode_step, decode_idx)

        except:
            print(x.shape)

        if decode_idx is None:
            # Teacher-forced pass over a full clip: position 0 has no causal
            # predecessor to predict it from, and the PhyCell branch below must
            # not see a position's own token before predicting it. Drop the
            # boundary position on both sides (matches the original SkyGPT).
            h_1 = h_1[:, 1:]

        # print(h_1.shape, h.shape)
        h_full = h[:, :-1] if x_full is None else self.fc_in(x_full)

        h_full = h_full.permute(0, 1, 4, 2, 3)
        # h_full = [batch, time, channel, H, W]
        h_0 = self.phycell(h_full)
        h_0 = h_0.permute(0, 1, 3, 4, 2)
        # h_0 = [batch, time, H, W, channel] -- channel-last, matching h_1's
        # layout, so decode_idx (a T,H,W triple) indexes the right dims below.

        if decode_idx is not None:
            h_0 = h_0[:, decode_idx[0], decode_idx[1], decode_idx[2], :]
            h_0 = h_0.reshape_as(h_1)

        h = h_0 + h_1

        h_out  = self.norm(h, cond)

        logits = self.fc_out(h_out)

        logits = logits.permute(0, 4, 1, 2, 3)
        return logits