# Adapted from SkyGPT (github.com/yuhao-nie/SkyGPT, MIT), which builds on VideoGPT (github.com/wilson1yan/VideoGPT, MIT). See NOTICE.
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.amp import GradScaler,autocast
from torch.utils.data import DataLoader, SubsetRandomSampler

from models.SkyGPT_utils.gpt import Skygpt # Santa
# from models.SkyGPT_utils.gpt2 import Skygpt # Aj Gap

from models.SkyGPT_utils.constrain_moments import K2M
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from models.common.unet_regressor import Model as Unet_regressor
from models.common.frame_export import dump_manifest_frames
import pdb

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        model_name = 'Skygpt'

        self.is_train = configs["Training"]["is_train"]

        self.image_channels = configs["VQVAE"]["Network"]["image_channels"]
        self.image_resize   = configs["ImageSequence"]["image_resize"]

        self.seq_length  = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]

        self.which_model = "VQVAE" if configs["Training"]["train_only_VQVAE"] else "Skygpt"

        self.beta_reconstruction = configs["VQVAE"]["Learning"]["beta_reconstruction"]
        self.beta_commitment     = configs["VQVAE"]["Learning"]["beta_commitment"]

        self.n_cond_frames = configs["Skygpt"]["Network"]["n_cond_frames"]
        # Iclr channel index in feature_x/feature_y ([h,m,doy,iclr] by convention) --
        # used by forecast() to reconstruct k_origin/SPM, same as models/PhyDNet.py's
        # self.iclr_idx (there read from Eval.iclr_idx; SkyGPT has no Eval block, so this
        # reads from Skygpt.Network instead, alongside n_cond_frames/n_ensemble/subset_size).
        self.iclr_idx = configs["Skygpt"]["Network"].get("iclr_idx", 3)

        dev    = "cuda:%d" % configs["Training"]["gpu_id"]
        device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.device = device
        self.scaler = GradScaler()

        self.model = Skygpt(configs).to(device)
        
        configs["Regressor"]["image_resize"] = configs["ImageSequence"]["image_resize"]
        self.regressor = Unet_regressor(configs["Regressor"]) 
        self.regressor.model.load_state_dict(torch.load(configs["Regressor"]["ckpt_path"], weights_only=False)['state_dict'])
        self.regressor.model.eval()
        # dir(self.model)
        # self.model.vqvae
        # self.model.transformer.attn_stack
        # self.model.transformer.fc_in

        if configs["Training"]["is_train"]:

            self.subsampling_enable = configs["Training"]["subsampling_enable"]
            self.subsampling_rate   = configs["Training"]["subsampling_rate"]


            if self.which_model == "VQVAE":
                for p in self.model.vqvae.parameters():
                    p.requires_grad = True

                self.optimizer = torch.optim.Adam(
                    self.model.vqvae.parameters(),
                    lr=eval(configs["VQVAE"]["Learning"]["learning_rate"]),
                    betas=eval(configs["VQVAE"]["Learning"]["learning_betas"]),
                )

                # experimental : Cosine-annealing LR scheduler 
                epoches = configs["Training"].get("epoches", 100)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epoches,
                    eta_min=1e-5,
                )

                self.num_param = sum(p.numel() for p in self.model.vqvae.parameters() if p.requires_grad)
                print("[%s] Num. params = %d" % (self.which_model, self.num_param))
                print('[%s] is_train = %s' % (self.which_model, self.is_train))

            else:  # Skygpt
                
                for p in self.model.transformer.parameters():
                    p.requires_grad = True
                 
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.transformer.parameters()),
                    lr=eval(configs["Skygpt"]["Learning"]["learning_rate"]),
                    betas=eval(configs["Skygpt"]["Learning"]["learning_betas"]),
                )

                # Matches the original SkyGPT's configure_optimizers(): CosineAnnealingLR
                # over the TOTAL number of training steps (not epochs), stepped once per
                # batch. Steps-per-epoch isn't known until the DataLoader is built inside
                # train(), so the scheduler is constructed lazily there on the first call.
                self.scheduler = None
                self._skygpt_total_epochs = configs["Training"]["epochs"]
 
                self.num_param = sum(p.numel() for p in self.model.transformer.parameters() if p.requires_grad)
                print("[%s] Num. params = %d" % (self.which_model, self.num_param))
                print('[%s] is_train = %s' % (self.which_model, self.is_train))
        else:
            self.num_param = sum(p.numel() for p in self.model.transformer.parameters() if p.requires_grad)
            print("[%s] Num. params = %d" % (self.which_model, self.num_param))
            print('[%s] is_train = %s' % (self.which_model, self.is_train))
    # ----------------------------------------------------------
    def forward(self, input):
        x_recon, vq_output = self.model(input)
        return x_recon, vq_output

    def _infer_target_frame(self, input):
        return self(input)

    def inference_target_sequence(self, input):
        B, P, C, H, W = input.shape
        self.model.eval()
        with torch.no_grad():
            x_recon, vq_output = self._infer_target_frame(input)
        B_pred, P_pred, C_pred, H_pred, W_pred = x_recon.shape
        assert B == B_pred
        assert P_pred == input.shape[1]
        assert C_pred == self.image_channels
        assert H == H_pred
        assert W == W_pred
        return x_recon, vq_output

    # ----------------------------------------------------------
    def update_schedule(self, epoch):
        """Commitment-loss weight is fixed (matches original SkyGPT, no warmup)."""
        self.epoch = epoch

        print(colored(f"[Schedule] EP {epoch:03d} | β_commit={self.model.vqvae.codebook.beta_commitment:.4f} | "
                      f"LR={self.optimizer.param_groups[0]['lr']:.2e}", 'cyan'))

    # ----------------------------------------------------------
    def vqvae_learning_loss(self, X_image, X_recon, vq_output):
        """Total loss = β_rec·MSE + commitment loss (matches original SkyGPT VQVAE loss)."""
        recon_loss      = F.mse_loss(X_recon, X_image)
        commitment_loss = vq_output["commitment_loss"]
        total_loss = self.beta_reconstruction * recon_loss + commitment_loss
        return total_loss

    # ----------------------------------------------------------
    def skygpt_learning_loss(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        k2m = K2M([self.model.transformer.kernel_size, self.model.transformer.kernel_size]).to(targets.device)
        for b in range(self.model.transformer.phycell.cell_list[0].input_dim):
            filters = self.model.transformer.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
            m = k2m(filters.double()).float()
            loss += self.model.transformer.constraint_criterion(m, self.model.transformer.constraints.to(targets.device))
        return loss

    def _per_sample_lpips(self, lpips_loss_fn, prediction_uint8, target_uint8):
        """Per-sample LPIPS. For a (B,P,C,H,W) input the per-frame axis P is kept
        (shape [B,P]), not collapsed, so callers can also derive per-horizon stats."""
        # LPIPS (AlexNet) expects inputs in [-1, 1]. uint8 [0,255] -> [-1,1]
        # (was /255.0 -> [0,1], which is the wrong range and inflates LPIPS;
        #  this matches the [-1,1] convention used for the joint model's LPIPS)
        prediction = prediction_uint8.float() / 255.0 * 2.0 - 1.0
        target = target_uint8.float() / 255.0 * 2.0 - 1.0

        if prediction.dim() == 5:
            B, P, C, H, W = prediction.shape
            prediction = prediction.reshape(B * P, C, H, W)
            target = target.reshape(B * P, C, H, W)
            lpips_raw = lpips_loss_fn(prediction, target)
            lpips_loss_fn.reset()
            return lpips_raw.reshape(B, P)
        elif prediction.dim() == 4:
            lpips_raw = lpips_loss_fn(prediction, target)
            lpips_loss_fn.reset()
            return lpips_raw.reshape(-1, 1)
        else:
            raise ValueError("Unsupported prediction shape %s for LPIPS" % (prediction.shape,))

    # ----------------------------------------------------------
    def train(self, dataset, config, epoch):

        
        
        # Skygpt used to precompute frozen-VQVAE encodings for the WHOLE
        # training split upfront (attribute_list=["...","vqvae"], convert_image2vqvae)
        # and hold them in CPU RAM. With the 12-frame clip (T'=3, up from T'=2) that
        # cache is ~57GB for this dataset's ~103k training samples, and OOM-killed the
        # process right as the precompute loop finished (confirmed via journalctl,
        # 2026-08-10). There is no correctness or speed benefit to precomputing --
        # the VQVAE is fully frozen/eval/no_grad either way, so on-the-fly per-batch
        # encoding (run_model's vqvae_target=None path, same as evaluation() already
        # does) produces identical results with no cache. Both branches now build the
        # same plain ["label","image"] dataset.
        full_train_dataset = dataset(config, self.device, mode="train",
                                 attribute_list=["label", "image"])



        if self.subsampling_enable:
            subset_size = int(self.subsampling_rate * len(full_train_dataset))
            indices     = torch.randperm(len(full_train_dataset))[:subset_size].tolist()
            sampler     = SubsetRandomSampler(indices)
            trainLoader = DataLoader(full_train_dataset,
                                     config["Training"]["batch_size"],
                                     sampler=sampler,
                                     shuffle=False,
                                     drop_last=True)
        else:
            trainLoader = DataLoader(full_train_dataset,
                                     config["Training"]["batch_size"],
                                     shuffle=True,
                                     drop_last=True)

        if self.which_model == "Skygpt" and self.scheduler is None:
            max_steps = self._skygpt_total_epochs * len(trainLoader)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps,
            )

        self.update_schedule(epoch)

        training_loss = []

        # Pooled over every element seen so far this epoch (matches evaluation()'s
        # Pixel-RMSE/PSNR/MAE) rather than averaged per-batch, so the epoch metric
        # doesn't depend on how batches were split.
        pixel_abserr_sum = 0.0
        pixel_sqerr_sum  = 0.0
        pixel_elem_count = 0

        if self.which_model == "VQVAE":
            self.model.vqvae.set_training(is_training=True)
            self.model.vqvae.train()
        else:
            self.model.transformer.set_training(is_training=True)
            self.model.transformer.train()
        
        pbar = tqdm(trainLoader)

        for input_index, data in enumerate(pbar): 

            
            X_image = Variable(data["image_x"].float() / 255).to(self.device) - 0.5
            Y_image = Variable(data["image_y"].float() / 255).to(self.device) - 0.5
            
            self.optimizer.zero_grad()

            if self.which_model == "VQVAE":
                # VQVAE trains on the 12-frame clip [image_x[-n_cond_frames:], image_y]
                # (4 real history + 8 real future) so it matches what Phase 2's frozen VQVAE
                # will be asked to encode -- NOT the raw 8-frame image_x alone (that would
                # train an 8-frame VQVAE and hand Phase 2 a clip length it never saw).
                # (Phase 2's own run_model()/convert_image2vqvae() callers need an analogous
                # fix before Phase 2 starts -- out of scope for this pass, which only covers
                # Phase 1.)
                X_image = torch.cat([X_image[:, -self.n_cond_frames:], Y_image], dim=1)  # (B, 12, C, H, W)
                with autocast(device_type='cuda'):
                    X_recon, vq_output = self.model.vqvae(X_image)
                    loss_total = self.vqvae_learning_loss(X_image, X_recon, vq_output)

                self.scaler.scale(loss_total).backward() # scale loss up then backward

                self.scaler.unscale_(self.optimizer) 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # unscale to appropiately clip gradient

                self.scaler.step(self.optimizer) # apply $$\theta_{new} = \theta_{old} - \eta \cdot \nabla L$$
                self.scaler.update() #
    
    
                current_perplexity = vq_output['perplexity'].item() 
                active_codes = len(torch.unique(vq_output['encodings'])) ## Calculate active codes


            else:
                # video must be the 12-frame clip [image_x[-n_cond_frames:],
                # image_y] -- run_model's frame_cond slice (video[:, :, :n_cond_frames])
                # expects `video`'s FIRST n_cond_frames to be the real conditioning
                # window, and its no-cache encode path (vqvae_target=None, see the
                # comment above full_train_dataset) needs the same 12-frame clip to
                # produce a T'=3 latent grid matching the Transformer's positional
                # embeddings. Passing raw 8-frame X_image would both put the OLDEST
                # frames into frame_cond and encode to the wrong T'.
                X_image = torch.cat([X_image[:, -self.n_cond_frames:], Y_image], dim=1)  # (B, 12, C, H, W)

                with autocast(device_type='cuda'):
                    Y_predict, embeded_predict, embeded_target = self.model.run_model(X_image)

                    # embeded_target :  [B, 3, 16, 16]  (T'=3, from the 12-frame clip)
                    # embeded_predict:  [B, 2, 16, 16]  (one latent step shorter, see run_model)
                    loss_total = self.skygpt_learning_loss(embeded_predict, embeded_target)

                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()  # per-step, matches the original's schedule

                # loss_total.backward()
                # self.optimizer.step()


                current_perplexity = None
                active_codes       = None
                

            # Clamp BEFORE computing PSNR  (original clamped after, inflating scores)
            X_image = torch.clamp(X_image, -0.5, 0.5) + 0.5
            if self.which_model == "VQVAE":
                X_recon = torch.clamp(X_recon, -0.5, 0.5) + 0.5
                X_image_uint8 = (X_image * 255).to(torch.uint8).detach()
                X_recon_uint8 = (X_recon * 255).to(torch.uint8).detach()
            else:
                # X_image was reassigned above to the 12-frame clip
                # [image_x[-n_cond_frames:], Y_image], so its tail
                # (X_image[:, -n_pred_frames:]) IS Y_image -- Y_predict is compared
                # against that tail, which is now genuinely the real future ground truth.
                n_pred_frames = Y_predict.shape[1]
                X_recon       = torch.clamp(Y_predict, -0.5, 0.5) + 0.5
                X_recon_uint8 = (X_recon * 255).to(torch.uint8).detach()
                X_image_uint8 = (X_image[:, -n_pred_frames:, :, :, :] * 255).to(torch.uint8).detach()

            # int16 diff avoids the uint8 wraparound that would silently corrupt
            # MAE/RMSE/PSNR (e.g. 0 - 1 -> 255 in unsigned arithmetic).
            pixel_diff = X_image_uint8.to(torch.int16) - X_recon_uint8.to(torch.int16)
            pixel_abserr_sum += pixel_diff.abs().sum().item()
            pixel_sqerr_sum  += (pixel_diff.float() ** 2).sum().item()
            pixel_elem_count += pixel_diff.numel()

            training_loss.append(loss_total.item())

            avg_loss = sum(training_loss) / len(training_loss)
            running_pixel_mse = pixel_sqerr_sum / pixel_elem_count
            avg_psnr = 100.0 if running_pixel_mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(running_pixel_mse))

            pplx_str = f"{current_perplexity:.1f}" if current_perplexity is not None else "N/A"
            pbar.set_description(
                f"Train EP:{epoch} [{input_index}] "
                f"Loss:{avg_loss:.4f} PSNR:{avg_psnr:.2f} "
                f"PPLX:{pplx_str} Active:{active_codes}"
            )

        if self.which_model == "VQVAE":
            # Step cosine scheduler once per epoch (VQVAE only -- Skygpt steps
            # per batch above, matching the original's schedule).
            self.scheduler.step()

        loss_per_epoch = sum(training_loss) / len(training_loss)
        pixel_mse = pixel_sqerr_sum / pixel_elem_count
        mae_per_epoch = pixel_abserr_sum / pixel_elem_count
        if pixel_mse == 0:
            rmse_per_epoch = 0.0
            psnr_per_epoch = 100.0  # log10(1/sqrt(0)) is undefined -- cap PSNR when MSE==0
        else:
            rmse_per_epoch = math.sqrt(pixel_mse)
            psnr_per_epoch = 20.0 * math.log10(255.0 / math.sqrt(pixel_mse))

        return {
            "loss_per_epoch": loss_per_epoch,
            "psnr_per_epoch": psnr_per_epoch,
            "rmse_per_epoch": rmse_per_epoch,
            "mae_per_epoch": mae_per_epoch,
            "lr":             self.optimizer.param_groups[0]['lr'],
        }

    # ----------------------------------------------------------
    def evaluation(self, dataset, config, epoch, mode="valid"):
        lpips_loss_fn = LPIPS(net_type='alex', reduction='none')
        lpips_loss_fn.to(self.device) # for evaluation

        if self.which_model == "VQVAE":
            self.model.vqvae.set_training(is_training=False)
            self.model.vqvae.eval()
        else:
            self.model.transformer.set_training(is_training=False)
            self.model.transformer.eval()

        valid_loss      = []
        samples_list    = []
        lpips_persample = []
        index_persample = []

        # Per-sample-per-horizon breakdowns -> [N,T] after concatenation, for the
        # PSNR_i/LPIPS_i/Reg[*]_i horizon stats (matches the pixel_sqerr_persample /
        # pixel_abserr_persample pattern in Text2Image_csi_condition_clean_only).
        pixel_sqerr_persample  = []
        pixel_abserr_persample = []

        # Regressor-based GHI error. VQVAE compares its full-window reconstruction
        # against the full label_x; Skygpt compares Y_predict against the matching
        # tail of label_x (see the branch below for why it's label_x, not label_y).
        reg_abserr_persample = []
        reg_sqerr_persample  = []

        # Pixel RMSE/PSNR/MAE are pooled over every element seen so far (matches
        # Text2Image_csi_condition_clean_only.run_evaluation) rather than averaged
        # per-sample, so the epoch metric doesn't depend on how batches were split.
        pixel_abserr_sum = 0.0
        pixel_sqerr_sum  = 0.0
        pixel_elem_count = 0
        sample_counter   = 0

        valid_dataset = dataset(config, self.device, mode=mode,
                                attribute_list=["label", "image"])

        validLoader   = DataLoader(valid_dataset,
                                   config["Training"]["%s_batch_size" % mode],
                                   shuffle=False, drop_last=False)

        pbar = tqdm(validLoader)

        for input_index, data in enumerate(pbar):
            X_image = Variable(data["image_x"].float() / 255).to(self.device) - 0.5

            if self.which_model == "VQVAE":
                # same 12-frame clip as train() -- see comment there.
                Y_image = Variable(data["image_y"].float() / 255).to(self.device) - 0.5
                X_image = torch.cat([X_image[:, -self.n_cond_frames:], Y_image], dim=1)  # (B, 12, C, H, W)

                with torch.no_grad():
                    X_recon, vq_output = self.model.vqvae(X_image)
                loss_total = self.vqvae_learning_loss(X_image, X_recon, vq_output)

                X_recon = torch.clamp(X_recon, -0.5, 0.5) + 0.5
                X_image = torch.clamp(X_image, -0.5, 0.5) + 0.5

                predictions_uint8 = (X_recon * 255).to(torch.uint8).detach()
                targets_uint8 = (X_image * 255).to(torch.uint8).detach()

                sample_preds  = predictions_uint8[0]
                sample_targs  = targets_uint8[0]

                # X_recon reconstructs the WHOLE 12-frame clip [image_x[-4:], image_y],
                # so the matching GHI ground truth is [label_x[-4:], label_y] -- not the
                # full label_x (that was the previous, 8-frame-only semantics).
                predictions_float = X_recon
                ghi_x = Variable(data["label_x"].float()).to(self.device)
                ghi_y = Variable(data["label_y"].float()).to(self.device)
                ghi_target = torch.cat([ghi_x[:, -self.n_cond_frames:], ghi_y], dim=1)[:, :, 0]

            else:
                # same 12-frame clip as the VQVAE branch above and train()'s
                # Skygpt branch -- run_model's frame_cond slice needs `video`'s first
                # n_cond_frames to be the real conditioning window, true only if
                # `video` is [image_x[-n_cond_frames:], image_y], not raw 8-frame
                # image_x.
                Y_image = Variable(data["image_y"].float() / 255).to(self.device) - 0.5
                X_image = torch.cat([X_image[:, -self.n_cond_frames:], Y_image], dim=1)  # (B, 12, C, H, W)

                with torch.no_grad():
                    Y_predict, embeded_predict, embeded_target = self.model.run_model(
                        X_image, is_training_attention=False)
                loss_total = self.skygpt_learning_loss(embeded_predict, embeded_target)

                n_pred_frames = Y_predict.shape[1]
                Y_predict = torch.clamp(Y_predict, -0.5, 0.5) + 0.5
                X_target = torch.clamp(X_image[:, -n_pred_frames:, :, :, :], -0.5, 0.5) + 0.5

                predictions_uint8 = (Y_predict * 255).to(torch.uint8).detach()
                targets_uint8 = (X_target * 255).to(torch.uint8).detach()

                sample_preds = predictions_uint8[0]
                sample_targs = targets_uint8[0]

                # X_image is now the 12-frame clip, so its tail
                # (X_image[:, -n_pred_frames:]) IS Y_image -- the matching GHI ground
                # truth is therefore label_y (that same window), not label_x.
                predictions_float = Y_predict
                ghi_y = Variable(data["label_y"].float()).to(self.device)
                ghi_target = ghi_y[:, -n_pred_frames:, 0]

            # Regressor-based GHI check (both branches): run the regressor on the
            # predicted frames and compare against the ground-truth irradiance for
            # those SAME frames (sliced above to match what each branch reconstructs).
            B_reg, T_reg, C_reg, H_reg, W_reg = predictions_float.shape  
            with torch.no_grad():
                estimated_irradiance = self.regressor(
                    predictions_float.reshape(B_reg * T_reg, C_reg, H_reg, W_reg) * 255)
            estimated_irradiance = estimated_irradiance.reshape(B_reg, T_reg)

            reg_abserr_persample.append(torch.abs(estimated_irradiance.detach() - ghi_target).cpu())
            reg_sqerr_persample.append(((estimated_irradiance.detach() - ghi_target) ** 2).cpu())

            # int16 diff avoids the uint8 wraparound that would silently corrupt
            # MAE/RMSE/PSNR (e.g. 0 - 1 -> 255 in unsigned arithmetic).
            pixel_diff = targets_uint8.to(torch.int16) - predictions_uint8.to(torch.int16)
            pixel_abserr_sum += pixel_diff.abs().sum().item()
            pixel_sqerr_sum  += (pixel_diff.float() ** 2).sum().item()
            pixel_elem_count += pixel_diff.numel()

            # per-sample-per-horizon pixel error (mean over C,H,W) -> [B,T], for the
            # PSNR_i / LPIPS_i horizon breakdown below.
            B_, T_ = predictions_uint8.shape[0], predictions_uint8.shape[1]
            _pdiff = pixel_diff.float().reshape(B_, T_, -1)
            pixel_sqerr_persample.append((_pdiff ** 2).mean(dim=2).cpu())
            pixel_abserr_persample.append(_pdiff.abs().mean(dim=2).cpu())

            lpips_batch = self._per_sample_lpips(lpips_loss_fn, predictions_uint8, targets_uint8)
            lpips_persample.append(lpips_batch.detach().cpu())

            index_persample.append(torch.arange(sample_counter, sample_counter + B_))
            sample_counter += B_

            valid_loss.append(loss_total.item())

            samples_list.append({
                "predictions": sample_preds,
                "ground_truth": sample_targs,
            })

            running_pixel_mse = pixel_sqerr_sum / pixel_elem_count
            running_psnr = 100.0 if running_pixel_mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(running_pixel_mse))
            running_lpips = torch.cat(lpips_persample).mean().item()

            tag = "Valid" if mode == "valid" else "Test"
            pbar.set_description(
                f"{tag} EP:{epoch} [{input_index}] "
                f"Loss:{sum(valid_loss)/len(valid_loss):.4f} "
                f"PSNR:{running_psnr:.2f} LPIPS:{running_lpips:.4f}"
            )

        pixel_mse = pixel_sqerr_sum / pixel_elem_count
        mae_per_epoch = pixel_abserr_sum / pixel_elem_count
        if pixel_mse == 0:
            rmse_per_epoch = 0.0
            psnr_per_epoch = 100.0  # log10(1/sqrt(0)) is undefined -- cap PSNR when MSE==0
        else:
            rmse_per_epoch = math.sqrt(pixel_mse)
            psnr_per_epoch = 20.0 * math.log10(255.0 / math.sqrt(pixel_mse))

        pixel_sqerr_all  = torch.cat(pixel_sqerr_persample, dim=0)   # [N, T]
        pixel_abserr_all = torch.cat(pixel_abserr_persample, dim=0)  # [N, T]
        lpips_all        = torch.cat(lpips_persample, dim=0)         # [N, T]
        index_all        = torch.cat(index_persample, dim=0)         # [N]

        lpips_per_epoch = lpips_all.mean().item()

        mse_per_horizon   = pixel_sqerr_all.mean(dim=0)   # [T] -- pooled MSE per horizon
        rmse_per_horizon  = torch.sqrt(mse_per_horizon)
        psnr_per_horizon  = 20.0 * torch.log10(255.0 / rmse_per_horizon.clamp(min=1e-8))
        mae_per_horizon   = pixel_abserr_all.mean(dim=0)
        lpips_per_horizon = lpips_all.mean(dim=0)

        n_frames = pixel_sqerr_all.shape[1]

        result = {
            "loss_per_epoch": sum(valid_loss) / len(valid_loss),
            "psnr_per_epoch": psnr_per_epoch,
            "rmse_per_epoch": rmse_per_epoch,
            "mae_per_epoch": mae_per_epoch,
            "lpips_per_epoch": lpips_per_epoch,
            "samples_list": samples_list,
        }

        for i in range(n_frames):
            result[f"PSNR_{i+1}"]  = psnr_per_horizon[i].item()
            result[f"LPIPS_{i+1}"] = lpips_per_horizon[i].item()

        reg_sqerr_all  = torch.cat(reg_sqerr_persample, dim=0)   # [N, T]
        reg_abserr_all = torch.cat(reg_abserr_persample, dim=0)  # [N, T]

        reg_rmse_per_horizon = torch.sqrt(reg_sqerr_all.mean(dim=0))
        reg_mae_per_horizon  = reg_abserr_all.mean(dim=0)

        for i in range(n_frames):
            result[f"Reg[RMSE]_{i+1}"] = reg_rmse_per_horizon[i].item()
            result[f"Reg[MAE]_{i+1}"]  = reg_mae_per_horizon[i].item()

        result["Reg[RMSE]_avg"] = reg_rmse_per_horizon.mean().item()
        result["Reg[MAE]_avg"]  = reg_mae_per_horizon.mean().item()

        # Per-sample stats dump for offline stratified analysis, test pass only --
        # mirrors Text2Image_csi_condition_clean_only.run_evaluation's CSI-eval npz
        # dump. csi_sqerr is omitted: SkyGPT has no Iclr/CSI ground truth to compare.
        if mode == "test":
            npz_file = os.path.join(config["ckpt_path"], "skygpt_reg_imgstats.npz")
            save_kwargs = dict(
                pixel_sqerr=pixel_sqerr_all.numpy(),     # [N, T]
                pixel_abserr=pixel_abserr_all.numpy(),   # [N, T]
                lpips=lpips_all.numpy(),                 # [N, T]
                reg_sqerr=reg_sqerr_all.numpy(),          # [N, T]
                reg_abserr=reg_abserr_all.numpy(),        # [N, T]
                index=index_all.numpy(),                 # [N]  join key (dataset order, shuffle=False)
                pixel_mse=np.float64(pixel_mse),          # pooled scalar (self-check)
                checkpoint=str(config["ckpt_path"]))

            np.savez_compressed(npz_file, **save_kwargs)
            print("[SkyGPT eval] saved per-sample image stats -> %s  ([%d,%d])"
                  % (npz_file, pixel_sqerr_all.shape[0], pixel_sqerr_all.shape[1]))

        return result

    # ----------------------------------------------------------
    def evaluation_sample(self, dataset, config, epoch, mode="valid"):
        """Same evaluation protocol as evaluation(), but drives predictions through
        Skygpt.sample() (autoregressive raster-scan generation, ~T'*H'*W' sequential
        forward passes per batch) instead of run_model() (one teacher-forced pass).
        Skygpt-only -- sample() has no VQVAE analogue, unlike evaluation().

        No loss is computed here: sample() returns decoded pixels only, not the
        logits/targets pair skygpt_learning_loss() needs, and (unlike evaluation())
        this method isn't wired into train.py's scheduler/early_stopping, so a
        missing loss is safe here.

        sample()'s raster scan regenerates tokens for the model's FULL input window
        (the conditioning span it's given as a cross-attention hint, plus the future
        span), not just the future tail -- so predictions are compared against the
        matching full X_image/label_x, not a tail slice.
        """
        assert self.which_model != "VQVAE", "evaluation_sample() is Skygpt-only (sample() has no VQVAE analogue)"

        lpips_loss_fn = LPIPS(net_type='alex', reduction='none')
        lpips_loss_fn.to(self.device)

        self.model.transformer.set_training(is_training=False)
        self.model.transformer.eval()

        samples_list    = []
        lpips_persample = []
        index_persample = []

        # Per-sample-per-horizon breakdowns -> [N,T] after concatenation, same
        # pattern as evaluation().
        pixel_sqerr_persample  = []
        pixel_abserr_persample = []

        # Regressor-based GHI error against the full window (predictions cover
        # the whole input window, so the matching ground truth is full label_x).
        reg_abserr_persample = []
        reg_sqerr_persample  = []

        pixel_abserr_sum = 0.0
        pixel_sqerr_sum  = 0.0
        pixel_elem_count = 0
        sample_counter   = 0

        valid_dataset = dataset(config, self.device, mode=mode,
                                attribute_list=["label", "image"])
        validLoader   = DataLoader(valid_dataset,
                                   config["Training"]["%s_batch_size" % mode],
                                   shuffle=False, drop_last=False)

        pbar = tqdm(validLoader)

        for input_index, data in enumerate(pbar):
            X_image = Variable(data["image_x"].float() / 255).to(self.device) - 0.5

            with torch.no_grad():
                Y_predict = self.model.sample(X_image.shape[0], X_image)

            Y_predict = torch.clamp(Y_predict, -0.5, 0.5) + 0.5
            X_target  = torch.clamp(X_image, -0.5, 0.5) + 0.5

            predictions_uint8 = (Y_predict * 255).to(torch.uint8).detach()
            targets_uint8 = (X_target * 255).to(torch.uint8).detach()

            sample_preds = predictions_uint8[0]
            sample_targs = targets_uint8[0]

            predictions_float = Y_predict
            ghi_x = Variable(data["label_x"].float()).to(self.device)
            ghi_target = ghi_x[:, :, 0]

            B_reg, T_reg, C_reg, H_reg, W_reg = predictions_float.shape
            with torch.no_grad():
                estimated_irradiance = self.regressor(
                    predictions_float.reshape(B_reg * T_reg, C_reg, H_reg, W_reg) * 255)
            estimated_irradiance = estimated_irradiance.reshape(B_reg, T_reg)

            reg_abserr_persample.append(torch.abs(estimated_irradiance.detach() - ghi_target).cpu())
            reg_sqerr_persample.append(((estimated_irradiance.detach() - ghi_target) ** 2).cpu())

            # int16 diff avoids the uint8 wraparound that would silently corrupt
            # MAE/RMSE/PSNR (e.g. 0 - 1 -> 255 in unsigned arithmetic).
            pixel_diff = targets_uint8.to(torch.int16) - predictions_uint8.to(torch.int16)
            pixel_abserr_sum += pixel_diff.abs().sum().item()
            pixel_sqerr_sum  += (pixel_diff.float() ** 2).sum().item()
            pixel_elem_count += pixel_diff.numel()

            B_, T_ = predictions_uint8.shape[0], predictions_uint8.shape[1]
            _pdiff = pixel_diff.float().reshape(B_, T_, -1)
            pixel_sqerr_persample.append((_pdiff ** 2).mean(dim=2).cpu())
            pixel_abserr_persample.append(_pdiff.abs().mean(dim=2).cpu())

            lpips_batch = self._per_sample_lpips(lpips_loss_fn, predictions_uint8, targets_uint8)
            lpips_persample.append(lpips_batch.detach().cpu())

            index_persample.append(torch.arange(sample_counter, sample_counter + B_))
            sample_counter += B_

            samples_list.append({
                "predictions": sample_preds,
                "ground_truth": sample_targs,
            })

            running_pixel_mse = pixel_sqerr_sum / pixel_elem_count
            running_psnr = 100.0 if running_pixel_mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(running_pixel_mse))
            running_lpips = torch.cat(lpips_persample).mean().item()

            tag = "Valid[sample]" if mode == "valid" else "Test[sample]"
            pbar.set_description(
                f"{tag} EP:{epoch} [{input_index}] "
                f"PSNR:{running_psnr:.2f} LPIPS:{running_lpips:.4f}"
            )

        pixel_mse = pixel_sqerr_sum / pixel_elem_count
        mae_per_epoch = pixel_abserr_sum / pixel_elem_count
        if pixel_mse == 0:
            rmse_per_epoch = 0.0
            psnr_per_epoch = 100.0  # log10(1/sqrt(0)) is undefined -- cap PSNR when MSE==0
        else:
            rmse_per_epoch = math.sqrt(pixel_mse)
            psnr_per_epoch = 20.0 * math.log10(255.0 / math.sqrt(pixel_mse))

        pixel_sqerr_all  = torch.cat(pixel_sqerr_persample, dim=0)   # [N, T]
        pixel_abserr_all = torch.cat(pixel_abserr_persample, dim=0)  # [N, T]
        lpips_all        = torch.cat(lpips_persample, dim=0)         # [N, T]
        index_all        = torch.cat(index_persample, dim=0)         # [N]

        lpips_per_epoch = lpips_all.mean().item()

        mse_per_horizon   = pixel_sqerr_all.mean(dim=0)   # [T] -- pooled MSE per frame position
        rmse_per_horizon  = torch.sqrt(mse_per_horizon)
        psnr_per_horizon  = 20.0 * torch.log10(255.0 / rmse_per_horizon.clamp(min=1e-8))
        mae_per_horizon   = pixel_abserr_all.mean(dim=0)
        lpips_per_horizon = lpips_all.mean(dim=0)

        n_frames = pixel_sqerr_all.shape[1]

        result = {
            "psnr_per_epoch": psnr_per_epoch,
            "rmse_per_epoch": rmse_per_epoch,
            "mae_per_epoch": mae_per_epoch,
            "lpips_per_epoch": lpips_per_epoch,
            "samples_list": samples_list,
        }

        for i in range(n_frames):
            result[f"PSNR_{i+1}"]  = psnr_per_horizon[i].item()
            result[f"LPIPS_{i+1}"] = lpips_per_horizon[i].item()

        reg_sqerr_all  = torch.cat(reg_sqerr_persample, dim=0)   # [N, T]
        reg_abserr_all = torch.cat(reg_abserr_persample, dim=0)  # [N, T]

        reg_rmse_per_horizon = torch.sqrt(reg_sqerr_all.mean(dim=0))
        reg_mae_per_horizon  = reg_abserr_all.mean(dim=0)

        for i in range(n_frames):
            result[f"Reg[RMSE]_{i+1}"] = reg_rmse_per_horizon[i].item()
            result[f"Reg[MAE]_{i+1}"]  = reg_mae_per_horizon[i].item()

        result["Reg[RMSE]_avg"] = reg_rmse_per_horizon.mean().item()
        result["Reg[MAE]_avg"]  = reg_mae_per_horizon.mean().item()

        # Per-sample stats dump for offline stratified analysis, test pass only.
        # Separate filename from evaluation()'s dump (skygpt_reg_imgstats.npz) so
        # the two evaluation protocols don't clobber each other in the same folder.
        if mode == "test":
            npz_file = os.path.join(config["ckpt_path"], "skygpt_sample_reg_imgstats.npz")
            save_kwargs = dict(
                pixel_sqerr=pixel_sqerr_all.numpy(),     # [N, T]
                pixel_abserr=pixel_abserr_all.numpy(),   # [N, T]
                lpips=lpips_all.numpy(),                 # [N, T]
                reg_sqerr=reg_sqerr_all.numpy(),          # [N, T]
                reg_abserr=reg_abserr_all.numpy(),        # [N, T]
                index=index_all.numpy(),                 # [N]  join key (dataset order, shuffle=False)
                pixel_mse=np.float64(pixel_mse),          # pooled scalar (self-check)
                checkpoint=str(config["ckpt_path"]))

            np.savez_compressed(npz_file, **save_kwargs)
            print("[SkyGPT sample-eval] saved per-sample image stats -> %s  ([%d,%d])"
                  % (npz_file, pixel_sqerr_all.shape[0], pixel_sqerr_all.shape[1]))

        return result

    # ----------------------------------------------------------
    def forecast(self, dataset, config, epoch, mode="test"):
        """Genuine forecast evaluation: given X_image as history, predict the
        truly-future, non-overlapping Y_image -- unlike evaluation_sample(),
        which compares sample()'s output against X_image's own tail.

        One sample() call can only ever produce seq_length frames total, split
        into n_cond_frames (given as conditioning) + the remainder (genuinely
        new) -- it has no notion of extrapolating further than that. To reach
        the full pred_length horizon, this chains multiple sample() calls in a
        sliding-window rollout: each call's newly-generated chunk becomes the
        next call's conditioning. The first call conditions on real history
        (X_image's last n_cond_frames frames); every call after that conditions
        on the model's own prior output, not ground truth, so errors can
        compound across rollout steps -- expected, not a bug.

        Also note: the model was trained with frame_cond = the first half of
        the SAME window being reconstructed (see run_model()), not "recent
        history preceding a disjoint future window" -- using it this way to
        reach Y_image is a repurposing of that mechanism, not something it was
        directly trained for.

        No loss (sample() has no logits/targets pair), Skygpt-only, and roughly
        (pred_length / (seq_length - n_cond_frames)) times the already-heavy
        per-call sample() cost -- meant for offline test-set evaluation, not
        per-epoch training validation (hence mode="test" default, unlike the
        other two methods above).
        """
        assert self.which_model != "VQVAE", "forecast() is Skygpt-only (sample() has no VQVAE analogue)"

        n_cond_frames = self.n_cond_frames
        chunk_frames  = self.seq_length - n_cond_frames
        assert chunk_frames > 0, "n_cond_frames must be < seq_length for forecast() to make progress"
        # The rollout below feeds each round's generated chunk straight back in
        # as the next round's conditioning window -- that only lines up with
        # what frame_cond expects (exactly n_cond_frames frames) if the chunk
        # size matches. Fails loudly here rather than silently truncating /
        # under-filling frame_cond if seq_length/n_cond_frames are ever changed.
        assert chunk_frames == n_cond_frames, (
            "forecast()'s rollout re-uses each round's generated chunk (size %d) directly as "
            "the next round's conditioning window, which requires seq_length - n_cond_frames "
            "== n_cond_frames (got %d vs %d)" % (chunk_frames, chunk_frames, n_cond_frames)
        )

        lpips_loss_fn = LPIPS(net_type='alex', reduction='none')
        lpips_loss_fn.to(self.device)

        self.model.transformer.set_training(is_training=False)
        self.model.transformer.eval()

        samples_list    = []
        lpips_persample = []
        index_persample = []

        # Per-sample-per-horizon breakdowns -> [N,T] after concatenation, same
        # pattern as evaluation()/evaluation_sample().
        pixel_sqerr_persample  = []
        pixel_abserr_persample = []

        # Regressor-based GHI error against the genuine future window -- the
        # whole point of forecast() is comparing against label_y, not label_x.
        reg_abserr_persample = []
        reg_sqerr_persample  = []
        # Signed error (estimated_irradiance - ghi_target), kept alongside the
        # abs/sq versions above so downstream analysis can compute a real mean
        # bias for the reg direction -- abs/sq alone cannot recover the sign.
        reg_signederr_persample = []

        # k_origin (clear-sky index at the forecast origin) and iclr_y (future clear-sky
        # irradiance) -- so SPM can be reconstructed straight from THIS npz, same as every
        # other model's forecast() (models/PhyDNet.py etc.). Previously absent entirely:
        # this npz had no k_origin/iclr_y at all, forcing downstream tools to point at a
        # DIFFERENT model's npz for the SPM reference.
        korg_persample = []
        iclr_persample = []

        pixel_abserr_sum = 0.0
        pixel_sqerr_sum  = 0.0
        pixel_elem_count = 0
        sample_counter   = 0

        # include "feature" so feature_x (Iclr at channel 3) is available for k_origin stratification
        valid_dataset = dataset(config, self.device, mode=mode,
                                attribute_list=["label", "image", "feature"])

        # --- SUBSET SELECTION (config Skygpt.Network.subset_indices / subset_size) ---
        # Prefer an explicit reference file over independent re-derivation -- the same
        # "prefer an explicit index file" pattern models/Xiong.py::_maybe_subset already uses
        # (Xiong being this repo's other genuine-ensemble model) -- so a re-run is
        # GUARANTEED byte-identical to a prior run regardless of numpy/RNG version drift,
        # not just "should reproduce given the same seed." Falls back to the original
        # stratified derivation (fixed seed 1234) only when no file is given -- e.g. the
        # very first time this subset is ever produced. `chosen` stays None when neither
        # path applies (full test set, no subsetting) -- guarded at save time below.
        chosen = None
        idx_file = config["Skygpt"]["Network"].get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            valid_dataset = torch.utils.data.Subset(valid_dataset, chosen)
            print("[SkyGPT] scoring the %d samples from %s (explicit reference file)"
                  % (len(chosen), path))
        else:
            subset_size = None
            try:
                subset_size = config["Skygpt"]["Network"].get("subset_size", None)
            except Exception:
                subset_size = None

            if subset_size is not None and int(subset_size) > 0 and int(subset_size) < len(valid_dataset):
                subset_size = int(subset_size)
                k_list = []
                for _i in range(len(valid_dataset)):
                    _d = valid_dataset[_i]
                    _ghi_i  = float(_d["label_x"][-1, 0])
                    _iclr_i = max(float(_d["feature_x"][-1, 3]), 1.0)   # feature_x = [h,m,doy,iclr]
                    k_list.append(_ghi_i / _iclr_i)
                k_arr = np.asarray(k_list)
                q = np.quantile(k_arr, [1/3, 2/3])
                bins = [np.where(k_arr < q[0])[0],
                        np.where((k_arr >= q[0]) & (k_arr < q[1]))[0],
                        np.where(k_arr >= q[1])[0]]
                rng = np.random.default_rng(1234)
                per = subset_size // 3
                chosen = []
                for b in bins:
                    take = min(per, len(b))
                    chosen.extend(rng.choice(b, size=take, replace=False).tolist())
                chosen = sorted(chosen)
                print("[SkyGPT] stratified subset: %d samples (per-regime ~%d), k tertiles %.3f/%.3f"
                      % (len(chosen), per, q[0], q[1]))
                valid_dataset = torch.utils.data.Subset(valid_dataset, chosen)

        # Always (re-)write the resolved indices next to THIS run's own npz, whichever path
        # produced them, so skygpt_subset_indices.npy always reflects what this run actually
        # scored (also duplicated inside the npz itself as `subset_indices`, see save below).
        if chosen is not None:
            np.save(os.path.join(config["ckpt_path"], "skygpt_subset_indices.npy"), np.asarray(chosen))

        validLoader   = DataLoader(valid_dataset,
                                   config["Training"]["%s_batch_size" % mode],
                                   shuffle=False, drop_last=False)

        pbar = tqdm(validLoader)

        # per-ensemble-member regressor GHI, accumulated for CRPS (small: [N, N_ens, T]).
        # Replaces the bulky HDF5 frame dump -- we keep only the GHI needed for the metric.
        ens_ghi_persample = []
        ghi_target_persample = []

        for input_index, data in enumerate(pbar):
            X_image = Variable(data["image_x"].float() / 255).to(self.device) - 0.5
            Y_image = Variable(data["image_y"].float() / 255).to(self.device) - 0.5

            # --- N-sample ENSEMBLE (config Skygpt.Network.n_ensemble, default 1) ---
            # n_ensemble=1 -> original single-sample rollout (unchanged).
            # n_ensemble>1 -> repeat the stochastic rollout N times; average frames for the
            # point forecast (fair vs the joint model's N=10), keep all N for CRPS.
            # Cost scales linearly with N -- use a subset for large N.
            _n_ens = 1
            try:
                _n_ens = int(config["Skygpt"]["Network"].get("n_ensemble", 1))
            except Exception:
                _n_ens = 1

            _ens_preds = []
            for _s in range(_n_ens):
                # Sliding-window autoregressive rollout -- see docstring.
                history = X_image[:, -n_cond_frames:, :, :, :]
                forecast_chunks = []
                frames_so_far = 0
                with torch.no_grad():
                    while frames_so_far < self.pred_length:
                        recon = self.model.sample(X_image.shape[0], history)
                        generated = recon[:, n_cond_frames:, :, :, :]
                        forecast_chunks.append(generated)
                        frames_so_far += generated.shape[1]
                        history = torch.clamp(generated, -0.5, 0.5)
                _yp = torch.cat(forecast_chunks, dim=1)[:, :self.pred_length, :, :, :]
                _ens_preds.append(_yp.unsqueeze(0))                 # [1,B,T,C,H,W]

            Y_samples = torch.cat(_ens_preds, dim=0)               # [N,B,T,C,H,W]
            Y_predict = Y_samples.mean(dim=0) if _n_ens > 1 else Y_samples[0]

            # --- per-member regressor GHI (for CRPS) ---------------------------------
            # Score EACH ensemble member's frames through the regressor -> GHI, so CRPS can
            # be computed offline from the small npz (no need to keep frames). Tiny array.
            if _n_ens > 1:
                _member_ghi = []                                    # each [B,T]
                for _m in range(_n_ens):
                    _ym = torch.clamp(Y_samples[_m], -0.5, 0.5) + 0.5   # [B,T,C,H,W] in [0,1]
                    _Bm, _Tm, _Cm, _Hm, _Wm = _ym.shape
                    with torch.no_grad():
                        _gm = self.regressor(_ym.reshape(_Bm * _Tm, _Cm, _Hm, _Wm) * 255)
                    _member_ghi.append(_gm.reshape(_Bm, _Tm).detach().cpu())
                ens_ghi_persample.append(torch.stack(_member_ghi, dim=1))   # [B, N_ens, T]

            Y_predict = torch.clamp(Y_predict, -0.5, 0.5) + 0.5
            Y_target  = torch.clamp(Y_image, -0.5, 0.5) + 0.5

            predictions_uint8 = (Y_predict * 255).to(torch.uint8).detach()
            targets_uint8 = (Y_target * 255).to(torch.uint8).detach()

            sample_preds = predictions_uint8[0]
            sample_targs = targets_uint8[0]

            predictions_float = Y_predict
            ghi_y = Variable(data["label_y"].float()).to(self.device)
            ghi_target = ghi_y[:, :, 0]
            if _n_ens > 1:
                ghi_target_persample.append(ghi_target.detach().cpu())   # [B,T], for CRPS

            # k_origin/iclr_y, straight from the same dataloader batch (already requested
            # via attribute_list=["label","image","feature"] above) -- last INPUT frame's
            # GHI/Iclr give the origin, feature_y gives Iclr over the forecast window. Same
            # formula as models/PhyDNet.py's forecast(): k_origin = GHI_i / Iclr_i.
            ghi_i    = data["label_x"][:, -1, 0].float()
            iclr_i   = data["feature_x"][:, -1, self.iclr_idx].float().clamp(min=1.0)
            iclr_y_b = data["feature_y"][:, :, self.iclr_idx].float().clamp(min=1.0)
            korg_persample.append((ghi_i / iclr_i).cpu())
            iclr_persample.append(iclr_y_b.cpu())

            B_reg, T_reg, C_reg, H_reg, W_reg = predictions_float.shape
            with torch.no_grad():
                estimated_irradiance = self.regressor(
                    predictions_float.reshape(B_reg * T_reg, C_reg, H_reg, W_reg) * 255)
            estimated_irradiance = estimated_irradiance.reshape(B_reg, T_reg)

            reg_abserr_persample.append(torch.abs(estimated_irradiance.detach() - ghi_target).cpu())
            reg_sqerr_persample.append(((estimated_irradiance.detach() - ghi_target) ** 2).cpu())
            reg_signederr_persample.append((estimated_irradiance.detach() - ghi_target).cpu())

            # int16 diff avoids the uint8 wraparound that would silently corrupt
            # MAE/RMSE/PSNR (e.g. 0 - 1 -> 255 in unsigned arithmetic).
            pixel_diff = targets_uint8.to(torch.int16) - predictions_uint8.to(torch.int16)
            pixel_abserr_sum += pixel_diff.abs().sum().item()
            pixel_sqerr_sum  += (pixel_diff.float() ** 2).sum().item()
            pixel_elem_count += pixel_diff.numel()

            B_, T_ = predictions_uint8.shape[0], predictions_uint8.shape[1]
            _pdiff = pixel_diff.float().reshape(B_, T_, -1)
            pixel_sqerr_persample.append((_pdiff ** 2).mean(dim=2).cpu())
            pixel_abserr_persample.append(_pdiff.abs().mean(dim=2).cpu())

            lpips_batch = self._per_sample_lpips(lpips_loss_fn, predictions_uint8, targets_uint8)
            lpips_persample.append(lpips_batch.detach().cpu())

            # Manifest-keyed frame export (aligned, unlike evaluation()'s shuffled
            # dumps). No-op for batches that don't overlap frame_manifest.csv, and a
            # no-op entirely unless that file is present. Written into THIS run's own
            # results folder (test-images/{case_id}/), not a shared path -- see
            # models/common/frame_export.py's module docstring.
            if mode == "test":
                dump_manifest_frames("skygpt", sample_counter, targets_uint8, predictions_uint8,
                                      out_dir=os.path.join(config["logging_version_path"], "test-images"))

            index_persample.append(torch.arange(sample_counter, sample_counter + B_))
            sample_counter += B_

            samples_list.append({
                "predictions": sample_preds,
                "ground_truth": sample_targs,
            })

            running_pixel_mse = pixel_sqerr_sum / pixel_elem_count
            running_psnr = 100.0 if running_pixel_mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(running_pixel_mse))
            running_lpips = torch.cat(lpips_persample).mean().item()

            tag = "Valid[forecast]" if mode == "valid" else "Test[forecast]"
            pbar.set_description(
                f"{tag} EP:{epoch} [{input_index}] "
                f"PSNR:{running_psnr:.2f} LPIPS:{running_lpips:.4f}"
            )

        pixel_mse = pixel_sqerr_sum / pixel_elem_count
        mae_per_epoch = pixel_abserr_sum / pixel_elem_count
        if pixel_mse == 0:
            rmse_per_epoch = 0.0
            psnr_per_epoch = 100.0  # log10(1/sqrt(0)) is undefined -- cap PSNR when MSE==0
        else:
            rmse_per_epoch = math.sqrt(pixel_mse)
            psnr_per_epoch = 20.0 * math.log10(255.0 / math.sqrt(pixel_mse))

        pixel_sqerr_all  = torch.cat(pixel_sqerr_persample, dim=0)   # [N, T]
        pixel_abserr_all = torch.cat(pixel_abserr_persample, dim=0)  # [N, T]
        lpips_all        = torch.cat(lpips_persample, dim=0)         # [N, T]
        index_all        = torch.cat(index_persample, dim=0)         # [N]

        lpips_per_epoch = lpips_all.mean().item()

        mse_per_horizon   = pixel_sqerr_all.mean(dim=0)   # [T] -- pooled MSE per forecast horizon step
        rmse_per_horizon  = torch.sqrt(mse_per_horizon)
        psnr_per_horizon  = 20.0 * torch.log10(255.0 / rmse_per_horizon.clamp(min=1e-8))
        mae_per_horizon   = pixel_abserr_all.mean(dim=0)
        lpips_per_horizon = lpips_all.mean(dim=0)

        n_frames = pixel_sqerr_all.shape[1]

        result = {
            "psnr_per_epoch": psnr_per_epoch,
            "rmse_per_epoch": rmse_per_epoch,
            "mae_per_epoch": mae_per_epoch,
            "lpips_per_epoch": lpips_per_epoch,
            "samples_list": samples_list,
        }

        for i in range(n_frames):
            result[f"PSNR_{i+1}"]  = psnr_per_horizon[i].item()
            result[f"LPIPS_{i+1}"] = lpips_per_horizon[i].item()

        reg_sqerr_all     = torch.cat(reg_sqerr_persample, dim=0)      # [N, T]
        reg_abserr_all    = torch.cat(reg_abserr_persample, dim=0)     # [N, T]
        reg_signederr_all = torch.cat(reg_signederr_persample, dim=0)  # [N, T]
        korg_all          = torch.cat(korg_persample, dim=0)           # [N]
        iclr_all          = torch.cat(iclr_persample, dim=0)           # [N, T]

        reg_rmse_per_horizon = torch.sqrt(reg_sqerr_all.mean(dim=0))
        reg_mae_per_horizon  = reg_abserr_all.mean(dim=0)
        reg_bias_per_horizon = reg_signederr_all.mean(dim=0)

        for i in range(n_frames):
            result[f"Reg[RMSE]_{i+1}"] = reg_rmse_per_horizon[i].item()
            result[f"Reg[MAE]_{i+1}"]  = reg_mae_per_horizon[i].item()
            result[f"Reg[Bias]_{i+1}"] = reg_bias_per_horizon[i].item()

        result["Reg[RMSE]_avg"] = reg_rmse_per_horizon.mean().item()
        result["Reg[MAE]_avg"]  = reg_mae_per_horizon.mean().item()
        result["Reg[Bias]_avg"] = reg_bias_per_horizon.mean().item()

        # Per-sample stats dump for offline stratified analysis, test pass only.
        # Separate filename from evaluation()'s/evaluation_sample()'s dumps so
        # the three evaluation protocols don't clobber each other.
        if mode == "test":
            npz_file = os.path.join(config["ckpt_path"], "skygpt_forecast_reg_imgstats.npz")
            save_kwargs = dict(
                pixel_sqerr=pixel_sqerr_all.numpy(),      # [N, T]
                pixel_abserr=pixel_abserr_all.numpy(),    # [N, T]
                lpips=lpips_all.numpy(),                  # [N, T]
                reg_sqerr=reg_sqerr_all.numpy(),          # [N, T]
                reg_abserr=reg_abserr_all.numpy(),        # [N, T]
                reg_signederr=reg_signederr_all.numpy(),  # [N, T]  signed: estimated_irradiance - ghi_target
                index=index_all.numpy(),                 # [N]  LOCAL 0..N-1 counter, NOT a join key
                k_origin=korg_all.numpy(),                # [N]  clear-sky index at forecast origin
                iclr_y=iclr_all.numpy(),                  # [N, T]  future clear-sky irradiance --
                                                           # reconstruct SPM as k_origin[:,None]*iclr_y,
                                                           # never a stored "SPM" field (buggy elsewhere
                                                           # in this repo -- see EVALUATION_HOWTO.md Sec.0)
                subset_indices=(np.asarray(chosen) if chosen is not None
                                 else np.arange(pixel_sqerr_all.shape[0])),  # [N] TRUE dataset-order
                                                           # join key (was previously only in the
                                                           # separate skygpt_subset_indices.npy file)
                pixel_mse=np.float64(pixel_mse),          # pooled scalar (self-check)
                checkpoint=str(config["ckpt_path"]))

            # per-ensemble-member regressor GHI + target GHI, for offline CRPS/FS.
            # Small: [N, N_ens, T]. Only present when n_ensemble > 1.
            if len(ens_ghi_persample) > 0:
                save_kwargs["ens_ghi"]    = torch.cat(ens_ghi_persample, dim=0).numpy()  # [N, N_ens, T]
                save_kwargs["ghi_target"] = torch.cat(ghi_target_persample, dim=0).numpy() # [N, T]
                # Aliases so the SAME generic [N,M,H]-ensemble tooling (analyze_ensemble.py,
                # which expects "S"/"GT") works on this npz unmodified, same as every other
                # genuine-ensemble/tiled-deterministic model's npz in this repo already does.
                save_kwargs["S"]  = save_kwargs["ens_ghi"]
                save_kwargs["GT"] = save_kwargs["ghi_target"]

            np.savez_compressed(npz_file, **save_kwargs)
            print("[SkyGPT forecast-eval] saved per-sample stats -> %s  ([%d,%d]%s)"
                  % (npz_file, pixel_sqerr_all.shape[0], pixel_sqerr_all.shape[1],
                     ", +ens_ghi for CRPS" if len(ens_ghi_persample) > 0 else ""))

        return result
