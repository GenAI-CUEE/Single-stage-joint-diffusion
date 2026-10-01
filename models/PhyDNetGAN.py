"""
models/PhyDNetGAN.py

PhyDNet+GAN image-sequence predictor (physics-informed recurrent generator +
LSGAN frame discriminator), ported from
yuhao-nie/SkyGPT/codes/video_prediction/PhyDNetGAN/models_v2.py +
constrain_moments.py (fetched into
SkyGPT_original/codes/video_prediction/PhyDNetGAN/ -- untracked reference
clone, same precedent as TSLib_original/ and Sunset_original/).

Deterministic model (confirmed against the reference authors' own evaluation
notebook, quan_eval_excld_VideoGPT_and_SkyGPT.ipynb: PhyDNet+GAN's own
predicted_images array -- labeled "SkyImageGAN" there -- is scored with plain
MSE/MAE/cosine-similarity, no ensemble/member dimension. The discriminator
only shapes the *training* loss (adversarial realism term); the generator
itself takes no noise input and produces one deterministic rollout at
inference). So forecast() follows the Group 3/Sunset deterministic M-tile
convention, not SkyGPT's genuine stochastic-sample ensembling -- see
see the module docstring above for the design rationale.

This is a separate file from models/PhyDNet.py on purpose: PhyDNet.py has the
discriminator wired in unconditionally (no way to get a plain, non-adversarial
PhyDNet from it today) and is being left as-is/deferred -- making plain PhyDNet
actually separable is a bigger job than this pass. PhyDNetGAN.py reuses the
shared, now-fixed building blocks in models/PhyDNet_utils/ (PhyCell/ConvLSTM
cell stacks, dcgan encoder/decoder blocks, Discr_frame, scheduling utils) --
duplicating ~500 lines of already-correct physics-cell code into a separate
PhyDNetGAN_utils/ folder would be pure churn for zero behavioural difference.

Two real bugs fixed relative to the pre-existing models/PhyDNet.py this was
modeled on (both in the shared utils, see their docstrings for detail):
  1. models/PhyDNet_utils/models_v2.py::EncoderRNN reprojected channels with a
     bogus 1x1 conv instead of recurrently rolling out into the future.
  2. models/PhyDNet_utils/{PhyDNet_utils,ConvLSTM_utils}.py's PhyCell/ConvLSTM
     stacks consumed a whole given sequence per call instead of the
     reference's one-timestep-at-a-time contract.
Also revived: models/PhyDNet_utils/scheduling_utils.py's
reserve_schedule_sampling_exp/schedule_sampling, imported but never called in
models/PhyDNet.py -- these are the reference's actual curriculum schedule for
teacher forcing on both the encode and decode phases, not optional polish.

    Model(configs)          built by main.py
      .num_param             logged by main.py
      .optimizer              Adam (generator)
      .discriminator_optimizer Adam (frame discriminator)
      .model                  EncoderRNN (nn.Module, the generator)
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats,
                                                          writes
                                                          phydnetgan_forecast_reg_imgstats.npz
"""

import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from models.common.eval_utils import PSNR_torch
from models.PhyDNet_utils.models_v2 import EncoderRNN
from models.PhyDNet_utils.gan_utils import Discr_frame
from models.PhyDNet_utils.scheduling_utils import reserve_schedule_sampling_exp, schedule_sampling

from losses.PhyDNet.losses import Generative_loss, Moment_regularization, Discriminative_loss
from models.common.metrics import forecast_skill, pooled_mae, pooled_rmse
from models.common.unet_regressor import Model as Unet_regressor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from models.common.frame_export import dump_manifest_frames


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        model_name = 'PhyDNetGAN'

        self.image_channels = configs["Network"]["image_channels"]
        self.image_resize = configs["ImageSequence"]["image_resize"]

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]

        self.iclr_idx = configs.get("Eval", {}).get("iclr_idx", 3)

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.device = device
        self.epoch = 0

        self.model = EncoderRNN(configs)
        self.model = self.model.to(device)

        self.discriminator = Discr_frame().to(device)

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("[%s] Num. params = %d" % (model_name, self.num_param))

        if configs["Training"]["is_train"]:

            self.epoch_enable_disc = configs["Learning"]["epoch_enable_disc"]
            self.beta_generative = configs["Learning"]["beta_generative"]
            self.beta_regularization = configs["Learning"]["beta_regularization"]
            self.beta_discrim = configs["Learning"]["beta_discrim"]

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs["Learning"]["learning_rate_gen"], betas=(0.5, 0.99))
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=configs["Learning"]["learning_rate_disc"], betas=(0.5, 0.99))

            self.generat_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.1)
            self.discrim_scheduler = ReduceLROnPlateau(self.discriminator_optimizer, mode='min', patience=5, factor=0.1)

            self.subsampling_enable = configs["Training"]["subsampling_enable"]
            self.subsampling_rate = configs["Training"]["subsampling_rate"]

            # Learning Losses
            self.generative_loss = Generative_loss(self.beta_generative).to(device)
            self.moment_regularization = Moment_regularization(self.beta_regularization).to(device)
            self.discriminative_loss = Discriminative_loss(self.beta_discrim).to(device)

        else:
            self.beta_generative = configs["Learning"]["beta_generative"]
            self.beta_regularization = configs["Learning"]["beta_regularization"]
            self.beta_discrim = configs["Learning"]["beta_discrim"]

            self.generative_loss = Generative_loss(self.beta_generative).to(device)
            self.moment_regularization = Moment_regularization(self.beta_regularization).to(device)
            self.discriminative_loss = Discriminative_loss(self.beta_discrim).to(device)
            # path = configs["ckpt_path"]
            # self.model.load_state_dict(torch.load(str(path)))
            # print('Infer mode: loaded %s model from %s' % (model_name, path))
            # self.model = self.model.eval()

            # Regressor(image -> GHI), needed by forecast() to score predicted
            # frames on the same GHI metric as every other model in the
            # comparison table. Same checkpoint/loading pattern as
            # models/SkyGPT.py -- generic, not SkyGPT-specific.
            if "Regressor" in configs:
                self.regressor = Unet_regressor(configs["Regressor"])
                self.regressor.model.load_state_dict(
                    torch.load(configs["Regressor"]["ckpt_path"], map_location="cpu", weights_only=False)['state_dict'])
                self.regressor.model.eval()

    def forward(self, input_image, y_target=None, teacher_forcing_ratio=0.0, encoder_sampling_ratio=1.0):
        return self.model(input_image, y_target, teacher_forcing_ratio, encoder_sampling_ratio)

    def update_schedule(self, epoch):
        self.epoch = epoch

    def learning_generative(self, Y_predict, Y_target, model):
        gen_loss = self.generative_loss(Y_predict, Y_target)
        regularization_loss = self.moment_regularization(model)
        loss = gen_loss + regularization_loss
        return loss

    def learning_discriminative(self, discr_frame_out_real, discr_frame_out_fake):
        loss = self.discriminative_loss(discr_frame_out_real, discr_frame_out_fake)
        return loss

    def train(self, dataset, config, epoch):

        full_train_dataset = dataset(config, self.device, mode="train", attribute_list=["label", "image"])

        self.update_schedule(epoch)

        if self.subsampling_enable:
            subset_size = int(self.subsampling_rate * len(full_train_dataset))
            indices = torch.randperm(len(full_train_dataset))[:subset_size].tolist()
            sampler = SubsetRandomSampler(indices)
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

        self.model.train()
        self.discriminator.train()

        # Scheduled sampling on BOTH encode and decode phases -- the reference
        # PhyDNetGAN.ipynb's actual curriculum (sampling_step_1=15,
        # sampling_step_2=30 epochs), previously imported but never called.
        encoder_sampling_ratio, _ = reserve_schedule_sampling_exp(epoch, self.seq_length - 1)
        teacher_forcing_ratio, _ = schedule_sampling(epoch, self.pred_length)

        psnr_blck_list = []
        training_gen_loss_list = []
        training_disc_loss_list = []

        pbar = tqdm(trainLoader)

        for input_index, data in enumerate(pbar):
            X_image = Variable(data["image_x"].float() / 255).to(self.device)
            Y_image = Variable(data["image_y"].float() / 255).to(self.device)

            if self.epoch >= self.epoch_enable_disc:

                self.optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                Y_predict = self.model(X_image, Y_image, teacher_forcing_ratio, encoder_sampling_ratio)
                gen_loss = self.learning_generative(Y_predict, Y_image, self.model)

                discr_frame_out_real = self.discriminator(Y_image.reshape(-1, self.image_channels, self.image_resize, self.image_resize)).view(-1)
                discr_frame_out_fake = self.discriminator(Y_predict.reshape(-1, self.image_channels, self.image_resize, self.image_resize)).view(-1)
                discrim_loss = self.learning_discriminative(discr_frame_out_real, discr_frame_out_fake)

                gen_loss.backward(retain_graph=True)
                discrim_loss.backward()

                self.optimizer.step()
                self.discriminator_optimizer.step()

            else:

                self.optimizer.zero_grad()

                Y_predict = self.model(X_image, Y_image, teacher_forcing_ratio, encoder_sampling_ratio)

                gen_loss = self.learning_generative(Y_predict, Y_image, self.model)

                gen_loss.backward()
                self.optimizer.step()

            psnr_BxC_temp = PSNR_torch(Y_predict * 255, Y_image * 255)
            psnr_BxC_temp = psnr_BxC_temp.view(-1).clone().detach()

            psnr_blck_list.append(psnr_BxC_temp.item())
            training_gen_loss_list.append(gen_loss.item())

            if self.epoch >= self.epoch_enable_disc:
                training_disc_loss_list.append(discrim_loss.item())
                pbar.set_description("Train: EP:%d [%d] Gen loss:%.4f Disc loss:%.4f PSNR:%.2f" % (epoch,
                                                                                                      input_index,
                                                                                                      sum(training_gen_loss_list) / len(training_gen_loss_list),
                                                                                                      sum(training_disc_loss_list) / len(training_disc_loss_list),
                                                                                                      sum(psnr_blck_list) / len(psnr_blck_list)))
            else:
                pbar.set_description("Train: EP:%d [%d] Gen loss:%.4f PSNR:%.2f" % (epoch,
                                                                                      input_index,
                                                                                      sum(training_gen_loss_list) / len(training_gen_loss_list),
                                                                                      sum(psnr_blck_list) / len(psnr_blck_list)))

        train_psnr_per_epoch = sum(psnr_blck_list) / len(psnr_blck_list)
        train_gen_loss_per_epoch = sum(training_gen_loss_list) / len(training_gen_loss_list)

        if self.epoch >= self.epoch_enable_disc:
            train_disc_loss_per_epoch = sum(training_disc_loss_list) / len(training_disc_loss_list)
            output_data = {"psnr_per_epoch": train_psnr_per_epoch,
                            "loss_per_epoch": train_gen_loss_per_epoch,
                            "disc_loss_per_epoch": train_disc_loss_per_epoch}
        else:
            output_data = {"psnr_per_epoch": train_psnr_per_epoch,
                            "loss_per_epoch": train_gen_loss_per_epoch,
                            "disc_loss_per_epoch": np.inf}

        return output_data

    def evaluation(self, dataset, config, epoch, mode="valid"):
        self.model.eval()
        self.discriminator.eval()

        psnr_blck = []
        valid_gen_loss = []
        valid_disc_loss = []
        samples_list = []

        valid_dataset = dataset(config, self.device, mode=mode, attribute_list=["label", "image"])
        validLoader = DataLoader(valid_dataset, config["Training"]["batch_size"], shuffle=True, drop_last=True)

        pbar = tqdm(validLoader)

        for input_index, data in enumerate(pbar):

            X_image = Variable(data["image_x"].float() / 255).to(self.device)
            Y_image = Variable(data["image_y"].float() / 255).to(self.device)

            with torch.no_grad():
                Y_predict = self.model(X_image)
                discr_frame_out_real = self.discriminator(Y_image.reshape(-1, self.image_channels, self.image_resize, self.image_resize)).view(-1)
                discr_frame_out_fake = self.discriminator(Y_predict.reshape(-1, self.image_channels, self.image_resize, self.image_resize)).view(-1)

            loss_total = self.learning_generative(Y_predict, Y_image, self.model)
            discrim_loss = self.learning_discriminative(discr_frame_out_real, discr_frame_out_fake)

            Y_predict_uint8 = (Y_predict * 255).to(torch.uint8).detach()
            Y_image_uint8 = (Y_image * 255).to(torch.uint8).detach()

            psnr_BxC_temp = PSNR_torch(Y_predict_uint8, Y_image_uint8)
            psnr_BxC_temp = psnr_BxC_temp.view(-1).clone().detach()

            psnr_blck.append(psnr_BxC_temp.item())
            valid_gen_loss.append(loss_total.item())
            valid_disc_loss.append(discrim_loss.item())

            if input_index < 5:
                samples_list.append({"predictions": Y_predict_uint8[0, :, :, :], "ground_truth": Y_image_uint8[0, :, :, :]})

            pbar.set_description("Valid: EP:%d [%d] Valid loss:%.4f Valid disc loss:%.4f  PSNR:%.2f" % (epoch, input_index, sum(valid_gen_loss) / (len(valid_gen_loss)), sum(valid_disc_loss) / (len(valid_disc_loss)), sum(psnr_blck) / (len(psnr_blck))))

        valid_psnr_per_epoch = sum(psnr_blck) / len(psnr_blck)
        valid_gen_loss_per_epoch = sum(valid_gen_loss) / len(valid_gen_loss)
        valid_disc_loss_per_epoch = sum(valid_disc_loss) / len(valid_disc_loss)

        output_data = {"psnr_per_epoch": valid_psnr_per_epoch,
                        "loss_per_epoch": valid_gen_loss_per_epoch,
                        "disc_loss_per_epoch": valid_disc_loss_per_epoch,
                        "samples_list": samples_list}
        return output_data

    def _maybe_subset(self, test_dataset, config, ev):
        """Ported from models/DLinear.py::_maybe_subset -- prefer an explicit
        index file, which is the only way to guarantee the SAME samples as
        SkyGPT/Xiong/Group 3."""
        idx_file = ev.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print("[PhyDNetGAN] scoring the %d samples from %s (matched to SkyGPT)"
                  % (len(chosen), path))
            return Subset(test_dataset, chosen), chosen
        return test_dataset, None

    def forecast(self, dataset, config, epoch, mode="test"):
        """The comparison path: autoregressively predicts the genuinely-future
        Y_image (no teacher forcing, no encoder scheduled sampling), scores it
        through the frozen image->GHI regressor (same one models/SkyGPT.py
        uses), then reports pooled per-horizon RMSE/MAE, FS vs. Smart
        Persistence (SPM), and CRPS := MAE (the exact identity for a
        deterministic forecast).
        """
        ev = config.get("Eval", {})

        self.model.eval()
        lpips_loss_fn = LPIPS(net_type="alex", reduction="none").to(self.device)

        test_dataset_full = dataset(config, self.device, mode=mode,
                                     attribute_list=["label", "feature", "image"])
        test_dataset, chosen = self._maybe_subset(test_dataset_full, config, ev)

        testLoader = DataLoader(test_dataset, config["Training"]["test_batch_size"],
                                 shuffle=False, drop_last=False)

        preds, gts, korgs, iclrs = [], [], [], []
        pixel_sqerr_list, pixel_abserr_list, lpips_list = [], [], []
        samples_list = []
        frame_row = 0  # row counter into the shared subset (frame_manifest.csv)

        pbar = tqdm(testLoader)
        for input_index, data in enumerate(pbar):
            X_image = Variable(data["image_x"].float() / 255).to(self.device)
            Y_image = data["image_y"].float() / 255

            ghi_target = data["label_y"][:, :, 0].float()
            ghi_i = data["label_x"][:, -1, 0].float()
            iclr_i = data["feature_x"][:, -1, self.iclr_idx].float().clamp(min=1.0)
            iclr_y = data["feature_y"][:, :, self.iclr_idx].float().clamp(min=1.0)

            with torch.no_grad():
                Y_predict = self.model(X_image).cpu()

                B_reg, T_reg, C_reg, H_reg, W_reg = Y_predict.shape
                estimated_irradiance = self.regressor(
                    (Y_predict.reshape(B_reg * T_reg, C_reg, H_reg, W_reg) * 255).to(self.device))
                estimated_irradiance = estimated_irradiance.reshape(B_reg, T_reg).cpu()

                # Image-quality metrics (LPIPS / pixel error), same convention as
                # models/Text2ImageJoint.py::forecast() -- uint8 [0,255] pixel diff
                # taken in int16 (avoids uint8 wraparound) for pixel_sqerr/
                # pixel_abserr, AlexNet LPIPS on [-1,1]-scaled frames. This model is
                # fully deterministic (one frame per sample, no ensembling), so these
                # are genuine per-sample values, not a tiled placeholder like the GHI
                # arrays below.
                Y_predict_u8 = (Y_predict * 255).to(torch.uint8)
                Y_image_u8 = (Y_image * 255).to(torch.uint8)
                pdiff = Y_image_u8.to(torch.int16) - Y_predict_u8.to(torch.int16)
                _pd = pdiff.float().reshape(B_reg, T_reg, -1)
                pixel_sqerr_batch = (_pd ** 2).mean(dim=2)
                pixel_abserr_batch = _pd.abs().mean(dim=2)

                pred_m11 = Y_predict.to(self.device) * 2 - 1
                targ_m11 = Y_image.to(self.device) * 2 - 1
                lpips_val = lpips_loss_fn(targ_m11.reshape(-1, *targ_m11.shape[2:]),
                                           pred_m11.reshape(-1, *pred_m11.shape[2:]))
                lpips_loss_fn.reset()
                lpips_batch = lpips_val.view(B_reg, T_reg).detach().cpu()

            preds.append(estimated_irradiance)
            gts.append(ghi_target)
            korgs.append(ghi_i / iclr_i)
            iclrs.append(iclr_y)
            pixel_sqerr_list.append(pixel_sqerr_batch)
            pixel_abserr_list.append(pixel_abserr_batch)
            lpips_list.append(lpips_batch)

            # Manifest-keyed frame export (aligned, unlike evaluation()'s shuffled
            # dumps). No-op for batches that don't overlap frame_manifest.csv, and a
            # no-op entirely unless that file is present. Written into THIS run's own
            # results folder (test-images/{case_id}/), not a shared path -- see
            # models/common/frame_export.py's module docstring.
            if mode == "test":
                dump_manifest_frames("phydnetgan", frame_row, Y_image_u8, Y_predict_u8,
                                      out_dir=os.path.join(config["logging_version_path"], "test-images"))
            frame_row += B_reg

            if input_index < 5:
                samples_list.append({
                    "predictions": (Y_predict[0] * 255).to(torch.uint8),
                    "ground_truth": (Y_image[0] * 255).to(torch.uint8)})

            pbar.set_description("Test[forecast] EP:%d [%d]" % (epoch, input_index))

        pred_all = torch.cat(preds, dim=0).numpy().astype(np.float64)    # [N,H]
        gt_all = torch.cat(gts, dim=0).numpy().astype(np.float64)        # [N,H]
        korg_all = torch.cat(korgs, dim=0).numpy().astype(np.float64)    # [N]
        iclr_all = torch.cat(iclrs, dim=0).numpy().astype(np.float64)    # [N,H]
        pixel_sqerr_all = torch.cat(pixel_sqerr_list, dim=0).numpy().astype(np.float64)    # [N,H]
        pixel_abserr_all = torch.cat(pixel_abserr_list, dim=0).numpy().astype(np.float64)  # [N,H]
        lpips_all = torch.cat(lpips_list, dim=0).numpy().astype(np.float64)                # [N,H]

        spm = korg_all[:, None] * iclr_all
        rmse = pooled_rmse(pred_all, gt_all)
        mae = pooled_mae(pred_all, gt_all)
        spm_rmse = pooled_rmse(spm, gt_all)
        spm_mae = pooled_mae(spm, gt_all)
        fs = forecast_skill(rmse, spm_rmse)
        crps = mae.copy()                      # deterministic: CRPS == MAE
        crpss = 1.0 - crps / spm_mae

        rmse_pooled = float(np.sqrt(np.mean(rmse ** 2)))
        spm_rmse_pooled = float(np.sqrt(np.mean(spm_rmse ** 2)))
        fs_pooled = 1.0 - rmse_pooled / spm_rmse_pooled

        # Image-quality pooled/per-horizon summaries -- same key names as
        # models/Text2ImageJoint.py::_forecast_metrics for the same physical
        # quantities (pooled pixel-RMSE, mean LPIPS, per-horizon LPIPS_%d).
        pixel_rmse_h = np.sqrt(pixel_sqerr_all.mean(axis=0))    # [H]
        lpips_h = lpips_all.mean(axis=0)                         # [H]
        pixel_rmse_pooled = float(np.sqrt(pixel_sqerr_all.mean()))
        lpips_mean = float(lpips_all.mean())

        result = {
            "loss_per_epoch": float(np.mean((pred_all - gt_all) ** 2)),
            "psnr_per_epoch": float(rmse.mean()),
            "samples_list": samples_list,
            "rmse_mean": float(rmse.mean()), "mae_mean": float(mae.mean()),
            "fs_mean": float(fs.mean()), "crps_mean": float(crps.mean()),
            "crpss_mean": float(crpss.mean()),
            "rmse_pooled": rmse_pooled, "fs_pooled": fs_pooled,
            "rmse_per_epoch": pixel_rmse_pooled, "lpips_per_epoch": lpips_mean,
        }
        for h in range(rmse.shape[0]):
            result["RMSE_%d" % (h + 1)] = float(rmse[h])
            result["MAE_%d" % (h + 1)] = float(mae[h])
            result["FS_%d" % (h + 1)] = float(fs[h])
            result["CRPSS_%d" % (h + 1)] = float(crpss[h])
            result["PixelRMSE_%d" % (h + 1)] = float(pixel_rmse_h[h])
            result["LPIPS_%d" % (h + 1)] = float(lpips_h[h])

        if mode == "test":
            M = int(ev.get("n_ensemble_tile", 100))
            # analyze_ensemble.py requires a genuine [N,M,H] ensemble array --
            # M identical copies of the point forecast gives exactly zero
            # spread; its own CRPS estimator then algebraically collapses to
            # exactly MAE. Same convention as every deterministic model in
            # this repo (DLinear/PatchTST/TimeXer/Sunset/ConvLSTM).
            S = np.tile(pred_all[:, None, :], (1, M, 1)).astype(np.float32)

            index_all = np.asarray(chosen) if chosen is not None else np.arange(gt_all.shape[0])

            # _reg_imgstats suffix (matching models/SkyGPT.py's own
            # skygpt_forecast_reg_imgstats.npz naming): flags that these GHI
            # numbers came from running the frozen regressor on predicted
            # IMAGES, not a native scalar head.
            npz_file = os.path.join(config["ckpt_path"], "phydnetgan_forecast_reg_imgstats.npz")
            np.savez_compressed(
                npz_file,
                S=S, GT=gt_all.astype(np.float32),
                k_origin=korg_all.astype(np.float32), iclr_y=iclr_all.astype(np.float32),
                SPM=spm.astype(np.float32),
                pred=pred_all.astype(np.float32),
                # --- image quality, same keys SkyGPT/Xiong/Text2ImageJoint write --- #
                pixel_sqerr=pixel_sqerr_all.astype(np.float32),
                pixel_abserr=pixel_abserr_all.astype(np.float32),
                lpips=lpips_all.astype(np.float32),
                index=index_all, subset_indices=index_all,
                checkpoint=str(config["ckpt_path"]),
                arm="phydnetgan")

            print("[PhyDNetGAN forecast-eval] saved -> %s  (N=%d, M=%d, H=%d)"
                  % (npz_file, S.shape[0], M, S.shape[2]))
            print("[PhyDNetGAN forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)
            print("[PhyDNetGAN forecast-eval] FS mean %+.3f  CRPSS mean %+.3f"
                  % (fs.mean(), crpss.mean()))
            print("[PhyDNetGAN forecast-eval] RMSE pooled %.2f  FS pooled %+.3f"
                  % (rmse_pooled, fs_pooled))
            print("[PhyDNetGAN forecast-eval] LPIPS mean %.4f  Pixel-RMSE pooled %.2f"
                  % (lpips_mean, pixel_rmse_pooled))

        return result
