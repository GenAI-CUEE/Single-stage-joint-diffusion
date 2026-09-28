"""
models/Xiong.py

Xiong et al. (2025) multimodal probabilistic solar forecasting, wrapped in the
same Model interface the rest of this repo uses (see models/SkyGPT.py):

    Model(configs)          built by main.py
      .num_param            logged by main.py
      .optimizer            wrapped by train.py's ReduceLROnPlateau
      .model                nn.Module; main.py's generic checkpoint path does
                            module.model.load_state_dict(ckpt['state_dict'])
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats

Paper: "Multimodal ultra-short-term probabilistic solar power forecasting with
generative AI and Transformer", Adv. Appl. Energy 20:100250.

THREE TRAINING STAGES, ONE CONFIG KEY
-------------------------------------
`Training.xiong_stage` plays the same role `Training.train_only_VQVAE` plays for
SkyGPT, except there are three values:

  "video"       Sec. 2.2   ConvGRU motion encoder + neural SDE + conditional
                           diffusion U-Net, trained on the eps loss (Eq. 8)
  "resnet"      Sec. 2.3.3 single-frame image -> irradiance regression on
                           noise+blur-degraded frames, bounded spectral norm
  "multimodal"  Sec. 2.3   Transformer over early-concatenated image/power
                           tokens + GPA head, MSE on mu (Eqs. 9-21)

Run them in that order, pointing each stage at the previous stage's checkpoint
directory via `Xiong.Network.warmstart_ckpt_path`. Because every stage lives
inside one `XiongNet`, the stage-3 checkpoint contains all three trained pieces,
so inference needs a single `version`.

forecast() IS THE COMPARISON PATH
---------------------------------
Like SkyGPT's, it targets the genuinely-future `label_y` and is the only method
whose numbers should appear in the paper. It reports the comparison in BOTH
framings the handoff notes require:

  (a) end-to-end forecast -- GHI straight from the GPA head, as a Gaussian
      mixture over N generated sky sequences (Eqs. 23-25). Comparable to the
      joint model's direct CSI head.
  (b) frames -> regressor -- the generated frames scored by the SAME frozen
      U-Net regressor that judges SkyGPT and the joint model. Comparable to
      SkyGPT's ens_ghi and to Reg[RMSE]/LPIPS/Pixel.

Xiong is the only baseline that has both a direct irradiance head and generated
frames, so it is the one model that can measure the extraction-pipeline penalty
directly, on itself: (a) vs (b) on identical samples.
"""

import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from models.common.unet_regressor import Model as Unet_regressor
from models.common.frame_export import dump_manifest_frames

from models.common.frame_utils import (build_target_transform, build_xcfg,
                                       cdm_like_degradation, normalize_frames,
                                       unpack_batch)
from models.common.metrics import (aggregate_mixture, crps_ensemble_fair,
                                        crps_mixture, mwis, pooled_mae,
                                        pooled_rmse, sample_from_mixture)
from models.Xiong_utils.multimodal import XiongNet

import pdb


STAGES = ("video", "resnet", "multimodal")


class Model(nn.Module):
    # ---------------------------------------------------------------- #
    def __init__(self, configs):
        super(Model, self).__init__()

        self.is_train = configs["Training"]["is_train"]
        self.stage = configs["Training"].get("xiong_stage", "multimodal")
        assert self.stage in STAGES, \
            "Training.xiong_stage must be one of %s (got %r)" % (STAGES, self.stage)

        # exp_valid.py guards forecast() on which_model != "VQVAE"; any non-VQVAE
        # string passes. Keep the stage in it so logs say which one ran.
        self.which_model = "Xiong-%s" % self.stage

        self.image_channels = configs["ImageSequence"].get("image_channels", 3)
        self.image_resize = configs["ImageSequence"]["image_resize"]
        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.xcfg = build_xcfg(configs)
        self.target = build_target_transform(self.xcfg)
        self.model = XiongNet(self.xcfg).to(self.device)

        # Same frozen judge SkyGPT and the joint model use, so frames->GHI
        # numbers are directly comparable across all three methods.
        configs["Regressor"]["image_resize"] = configs["ImageSequence"]["image_resize"]
        self.regressor = Unet_regressor(configs["Regressor"])
        try:
            self.regressor.model.load_state_dict(
                torch.load(configs["Regressor"]["ckpt_path"], map_location=self.device, weights_only=False)["state_dict"])
        except RuntimeError as e:
            # The regressor's final 1x1 conv is sized H*W (64x64 -> 4096 input
            # channels), so the checkpoint only fits at the resolution it was
            # trained on. The line above propagates ImageSequence.image_resize
            # into the regressor, so changing the image size anywhere in the
            # config silently re-shapes this network. Raw torch reports it as a
            # bare shape mismatch, which does not point at the cause.
            raise RuntimeError(
                "Regressor checkpoint does not fit the network built from this "
                "config.\n  ImageSequence.image_resize = %s  (copied into "
                "Regressor.image_resize)\n  ckpt_path = %s\n"
                "The shipped checkpoint expects image_resize = 64 "
                "(conv_final has 64*64 = 4096 input channels; at 32 it would be "
                "1024). Restore image_resize to 64, or point ckpt_path at a "
                "regressor trained at this resolution.\n\noriginal error: %s"
                % (configs["ImageSequence"]["image_resize"],
                   configs["Regressor"]["ckpt_path"], e))
        self.regressor.model.eval()

        self._warmstart(configs)
        self.model.set_stage(self.stage)

        # cached datasets: unlike SkyGPT's VQVAE-token dataset, nothing here
        # depends on model state, so rebuilding it every epoch is pure overhead
        self._ds_cache = {}

        if self.is_train:
            self.subsampling_enable = configs["Training"]["subsampling_enable"]
            self.subsampling_rate = configs["Training"]["subsampling_rate"]

            learn = configs["Xiong"]["Learning"][self.stage]
            params = [p for p in self.model.stage_parameters(self.stage) if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params,
                lr=eval(learn["learning_rate"]) if isinstance(learn["learning_rate"], str)
                else learn["learning_rate"],
                betas=eval(learn["learning_betas"]) if isinstance(learn.get("learning_betas"), str)
                else tuple(learn.get("learning_betas", (0.9, 0.999))),
                weight_decay=learn.get("weight_decay", 0.0))
            self.grad_clip = learn.get("grad_clip", 1.0)
            self.amp = learn.get("amp", True) and self.device.type == "cuda"

            # train.py owns a ReduceLROnPlateau over self.optimizer; nothing extra here.
            self.scheduler = None
            self.num_param = sum(p.numel() for p in params)
        else:
            self.grad_clip, self.amp = 1.0, False
            self.num_param = sum(p.numel() for p in self.model.parameters())

        print("[%s] Num. params = %d" % (self.which_model, self.num_param))
        print("[%s] is_train = %s" % (self.which_model, self.is_train))
        print("[%s] %s" % (self.which_model, self.model.forecaster.summary()))
        print("[%s] target parameterisation = %s"
              % (self.which_model, self.xcfg["target"]["mode"]))

    # ---------------------------------------------------------------- #
    # cross-stage warm start
    # ---------------------------------------------------------------- #
    def _warmstart(self, configs):
        """
        Load whatever a previous stage already trained. `warmstart_ckpt_path` is
        relative to Main_logging_path, exactly like SkyGPT's `vaque_ckp_path`.

        Stage 2 needs stage 1's video predictor; stage 3 needs both. Since every
        checkpoint stores the whole XiongNet, one path is enough for stage 3 as
        long as it points at the stage-2 run (which itself warm-started from
        stage 1). Missing/extra keys are loaded non-strictly and reported.
        """
        rel = configs.get("Xiong", {}).get("Network", {}).get("warmstart_ckpt_path", None)
        if not rel:
            if self.stage != "video":
                print(colored("[Xiong] no warmstart_ckpt_path set -- stage '%s' is starting "
                              "from RANDOM upstream weights. That is almost certainly not "
                              "what you want." % self.stage, "yellow"))
            return

        ckpt_dir = os.path.join(configs["Main_logging_path"], rel)
        weights = sorted(glob.glob("%s/*.pth" % ckpt_dir))
        if not weights:
            raise FileNotFoundError("[Xiong] warmstart_ckpt_path has no .pth: %s" % ckpt_dir)

        path = weights[-1]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        report = self.model.load_state_dict(sd, strict=False)
        print(colored("[Xiong] warm start from %s (ep %s)"
                      % (path, ckpt.get("epoch", "?")), "cyan"))
        print("        missing %d key(s), unexpected %d key(s)"
              % (len(report.missing_keys), len(report.unexpected_keys)))
        if report.missing_keys:
            print("        first missing: %s" % report.missing_keys[:3])

    # ---------------------------------------------------------------- #
    # small helpers
    # ---------------------------------------------------------------- #
    def _dataset(self, dataset, config, mode):
        """Cached dataset construction, with this repo's positional signature."""
        if mode not in self._ds_cache:
            self._ds_cache[mode] = dataset(config, self.device, mode=mode,
                                           attribute_list=["label", "image", "feature"])
        return self._ds_cache[mode]

    def _unpack(self, data):
        return unpack_batch(data, self.xcfg, self.device)

    @staticmethod
    def _to_unit(x):
        """model space [-1,1] -> [0,1], the range the regressor and uint8 expect."""
        return (torch.clamp(x, -1.0, 1.0) + 1.0) * 0.5

    def _regress(self, frames_model_space):
        """
        frames [B,T,C,H,W] in [-1,1] -> GHI [B,T] in W/m^2.

        Fed as *255 in [0,255], matching how SkyGPT and the joint model feed this
        same regressor. Any change here silently breaks cross-method comparability.
        """
        B, T, C, H, W = frames_model_space.shape
        x = self._to_unit(frames_model_space).reshape(B * T, C, H, W) * 255.0
        with torch.no_grad():
            ghi = self.regressor(x)
        return ghi.reshape(B, T)

    def _per_sample_lpips(self, lpips_loss_fn, prediction_uint8, target_uint8):
        """Per-sample LPIPS, [B,P] kept. uint8 [0,255] -> [-1,1] (the fixed scale)."""
        prediction = prediction_uint8.float() / 255.0 * 2.0 - 1.0
        target = target_uint8.float() / 255.0 * 2.0 - 1.0
        if prediction.dim() == 5:
            B, P, C, H, W = prediction.shape
            lpips_raw = lpips_loss_fn(prediction.reshape(B * P, C, H, W),
                                      target.reshape(B * P, C, H, W))
            lpips_loss_fn.reset()
            return lpips_raw.reshape(B, P)
        lpips_raw = lpips_loss_fn(prediction, target)
        lpips_loss_fn.reset()
        return lpips_raw.reshape(-1, 1)

    def _degraded_future(self, b):
        """
        Real future frames degraded to look CDM-generated (Sec. 2.3.3). This is
        what the image branch sees in stages 2 and 3; generated frames only
        appear at inference.
        """
        aug = self.xcfg["resnet"]["aug"]
        y = b["y_frames"]
        B, T = y.shape[:2]
        flat = cdm_like_degradation(y.reshape(B * T, *y.shape[2:]),
                                    aug["noise_std"], aug["blur_sigma_max"])
        return flat.reshape(B, T, *y.shape[2:])

    def update_schedule(self, epoch):
        self.epoch = epoch
        print(colored("[Schedule] EP %03d | stage=%s | LR=%.2e"
                      % (epoch, self.stage, self.optimizer.param_groups[0]["lr"]), "cyan"))

    # ================================================================== #
    # 1. train()
    # ================================================================== #
    def train(self, dataset, config, epoch):
        full_train_dataset = self._dataset(dataset, config, "train")

        if self.subsampling_enable:
            subset_size = int(self.subsampling_rate * len(full_train_dataset))
            indices = torch.randperm(len(full_train_dataset))[:subset_size].tolist()
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     sampler=SubsetRandomSampler(indices),
                                     shuffle=False, drop_last=True)
        else:
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     shuffle=True, drop_last=True)

        self.update_schedule(epoch)
        self.model.train()
        self.model.set_stage(self.stage)

        training_loss = []
        # pooled, not per-batch averaged -- per-batch averaging of a heavy-tailed
        # error understates it badly, which is exactly the artefact that inflated
        # an earlier FS number in this project
        sqerr_sum, abserr_sum, elem_count = 0.0, 0.0, 0
        pbar = tqdm(trainLoader)

        for input_index, data in enumerate(pbar):
            b = self._unpack(data)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                if self.stage == "video":
                    loss_total = self.model.video.loss(b["x_frames"], b["y_frames"])
                    pred_z = target_z = None

                elif self.stage == "resnet":
                    frames = torch.cat([b["x_frames"], b["y_frames"]], dim=1)
                    target_z = torch.cat([b["z_x"], b["z_y"]], dim=1)
                    B, T = target_z.shape
                    frames = cdm_like_degradation(
                        frames.reshape(B * T, *frames.shape[2:]),
                        self.xcfg["resnet"]["aug"]["noise_std"],
                        self.xcfg["resnet"]["aug"]["blur_sigma_max"])
                    pred_z = self.model.forecaster.resnet.forward_regress(frames)
                    target_z = target_z.reshape(B * T)
                    loss_total = F.mse_loss(pred_z.float(), target_z)

                else:  # multimodal
                    frames = self._degraded_future(b)
                    pred_z = self.model.forecaster(frames, b["z_x"], b["u_x"], b["u_y"])
                    target_z = b["z_y"]
                    loss_total = F.mse_loss(pred_z.float(), target_z)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.stage_parameters(self.stage) if p.requires_grad],
                max_norm=self.grad_clip)
            self.optimizer.step()

            training_loss.append(loss_total.item())

            if pred_z is not None:
                d = (pred_z.detach().float() - target_z).flatten()
                sqerr_sum += float((d ** 2).sum())
                abserr_sum += float(d.abs().sum())
                elem_count += d.numel()

            avg_loss = sum(training_loss) / len(training_loss)
            extra = ("RMSE(z):%.4f" % math.sqrt(sqerr_sum / elem_count)) if elem_count else ""
            pbar.set_description("Train[%s] EP:%d [%d] Loss:%.4f %s"
                                 % (self.stage, epoch, input_index, avg_loss, extra))

        loss_per_epoch = sum(training_loss) / len(training_loss)
        out = {
            "loss_per_epoch": loss_per_epoch,
            # train.py logs psnr_per_epoch unconditionally. Only the video stage
            # has a pixel-space PSNR to report; the scalar stages log 0.0 and
            # carry their real metric in rmse_per_epoch (normalised units).
            "psnr_per_epoch": 0.0,
            "rmse_per_epoch": math.sqrt(sqerr_sum / elem_count) if elem_count else 0.0,
            "mae_per_epoch": abserr_sum / elem_count if elem_count else 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return out

    # ================================================================== #
    # 2. evaluation()  -- convergence monitor, cheap on purpose
    # ================================================================== #
    def evaluation(self, dataset, config, epoch, mode="valid"):
        """
        Deliberately cheap: no DDIM sampling, no ensembling. Its only job is to
        tell train.py / EarlyStopping whether the current stage is still
        improving, so `loss_per_epoch` is the quantity to MINIMISE for the
        active stage:

          video       validation eps-MSE (Eq. 8) -- plus, every
                      `eval_sample_batches` batches, real sampled frames so the
                      image metrics and the tracker dumps aren't blind
          resnet      validation MSE in normalised target units
          multimodal  pooled GHI RMSE in W/m^2 on degraded real frames

        The multimodal number here is optimistic relative to forecast(): it sees
        degraded REAL future frames, not generated ones. That is the right
        monitor (it isolates the forecaster from video-model noise) but it is
        NOT the number to report. Report forecast().
        """
        self.model.eval()
        valid_dataset = self._dataset(dataset, config, mode)
        validLoader = DataLoader(valid_dataset,
                                 config["Training"]["%s_batch_size" % mode],
                                 shuffle=False, drop_last=False)

        max_batches = config["Xiong"]["Network"].get("eval_max_batches", None)
        n_sample_batches = config["Xiong"]["Network"].get("eval_sample_batches", 2)

        lpips_loss_fn = None
        if self.stage == "video":
            lpips_loss_fn = LPIPS(net_type="alex", reduction="none").to(self.device)

        valid_loss = []
        samples_list = []
        preds_all, gts_all = [], []
        sqerr_sum, abserr_sum, elem_count = 0.0, 0.0, 0
        pixel_sqerr_sum, pixel_abserr_sum, pixel_elem = 0.0, 0.0, 0
        lpips_persample = []

        pbar = tqdm(validLoader)
        for input_index, data in enumerate(pbar):
            if max_batches is not None and input_index >= int(max_batches):
                break
            b = self._unpack(data)

            with torch.no_grad():
                if self.stage == "video":
                    # Deterministic per batch: the eps loss samples a random
                    # frame index and diffusion step, so an unseeded validation
                    # loss is mostly sampling noise and EarlyStopping would fire
                    # on it. Same draws every epoch -> the curve is comparable.
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(20_000 + input_index)
                    loss_total = self.model.video.loss(b["x_frames"], b["y_frames"],
                                                       generator=gen)
                    valid_loss.append(loss_total.item())

                    if input_index < n_sample_batches:
                        gen = self.model.video.sample_sequence(
                            b["x_frames"], n_samples=1,
                            steps=config["Xiong"]["Network"].get("eval_ddim_steps", 20))[:, 0]
                        pu8 = (self._to_unit(gen) * 255).to(torch.uint8)
                        tu8 = (self._to_unit(b["y_frames"]) * 255).to(torch.uint8)
                        # int16 diff: uint8 arithmetic wraps (0 - 1 -> 255)
                        pdiff = tu8.to(torch.int16) - pu8.to(torch.int16)
                        pixel_sqerr_sum += float((pdiff.float() ** 2).sum())
                        pixel_abserr_sum += float(pdiff.abs().sum())
                        pixel_elem += pdiff.numel()
                        lpips_persample.append(
                            self._per_sample_lpips(lpips_loss_fn, pu8, tu8).cpu())
                        samples_list.append({"predictions": pu8[0], "ground_truth": tu8[0]})

                elif self.stage == "resnet":
                    frames = torch.cat([b["x_frames"], b["y_frames"]], dim=1)
                    z = torch.cat([b["z_x"], b["z_y"]], dim=1)
                    B, T = z.shape
                    frames = cdm_like_degradation(
                        frames.reshape(B * T, *frames.shape[2:]),
                        self.xcfg["resnet"]["aug"]["noise_std"],
                        self.xcfg["resnet"]["aug"]["blur_sigma_max"])
                    pred = self.model.forecaster.resnet.forward_regress(frames)
                    z = z.reshape(B * T)
                    valid_loss.append(float(F.mse_loss(pred, z)))
                    iclr = torch.cat([b["iclr_x"], b["iclr_y"]], dim=1).reshape(B * T)
                    ghi = torch.cat([b["ghi_x"], b["ghi_y"]], dim=1).reshape(B * T)
                    preds_all.append(self.target.decode_mean(pred, iclr).cpu())
                    gts_all.append(ghi.cpu())
                    d = (pred - z)
                    sqerr_sum += float((d ** 2).sum())
                    abserr_sum += float(d.abs().sum())
                    elem_count += d.numel()

                else:  # multimodal
                    frames = self._degraded_future(b)
                    mu = self.model.forecaster(frames, b["z_x"], b["u_x"], b["u_y"])
                    valid_loss.append(float(F.mse_loss(mu, b["z_y"])))
                    preds_all.append(self.target.decode_mean(mu, b["iclr_y"]).cpu())
                    gts_all.append(b["ghi_y"].cpu())
                    d = (mu - b["z_y"])
                    sqerr_sum += float((d ** 2).sum())
                    abserr_sum += float(d.abs().sum())
                    elem_count += d.numel()

            tag = "Valid" if mode == "valid" else "Test"
            pbar.set_description("%s[%s] EP:%d [%d] Loss:%.4f"
                                 % (tag, self.stage, epoch, input_index,
                                    sum(valid_loss) / len(valid_loss)))

        result = {"samples_list": samples_list,
                  "loss_per_epoch": sum(valid_loss) / max(len(valid_loss), 1)}

        if self.stage == "video":
            pixel_mse = pixel_sqerr_sum / max(pixel_elem, 1)
            result["psnr_per_epoch"] = (100.0 if pixel_mse == 0 else
                                        20.0 * math.log10(255.0 / math.sqrt(pixel_mse)))
            result["rmse_per_epoch"] = math.sqrt(pixel_mse)
            result["mae_per_epoch"] = pixel_abserr_sum / max(pixel_elem, 1)
            if lpips_persample:
                result["lpips_per_epoch"] = float(torch.cat(lpips_persample).mean())
        else:
            preds = torch.cat(preds_all).numpy().astype(np.float64)
            gts = torch.cat(gts_all).numpy().astype(np.float64)
            ghi_rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))   # pooled
            result["psnr_per_epoch"] = 0.0
            result["rmse_per_epoch"] = math.sqrt(sqerr_sum / elem_count)
            result["mae_per_epoch"] = abserr_sum / elem_count
            result["GHI[RMSE]"] = ghi_rmse
            if self.stage == "multimodal":
                # EarlyStopping should select on the metric the paper reports,
                # not on normalised MSE
                result["loss_per_epoch"] = ghi_rmse

        return result

    # ================================================================== #
    # 3. forecast()  -- THE test-set path; these are the numbers to report
    # ================================================================== #
    def forecast(self, dataset, config, epoch, mode="test"):
        """
        Full pipeline, N generated sky sequences per clip (Sec. 2.2), each scored
        by the multimodal Transformer + GPA head (Sec. 2.3), combined into one
        Gaussian mixture (Eqs. 23-25).

        Cost: N x pred_length x ddim_steps U-Net evaluations per clip. Use
        `Xiong.Network.subset_size` / `subset_indices` to bound it -- and prefer
        `subset_indices` pointing at skygpt_subset_indices.npy so this scores the
        EXACT samples SkyGPT and the joint model were scored on.
        """
        assert self.stage == "multimodal" or not self.is_train, \
            "forecast() needs all three stages trained; load a stage-3 checkpoint"

        net = config["Xiong"]["Network"]
        n_ens = int(net.get("n_ensemble", 10))
        ddim_steps = int(net.get("forecast_ddim_steps", self.xcfg["video"]["ddim_steps"]))
        iclr_floor = self.xcfg["target"]["iclr_floor"]

        self.model.eval()
        lpips_loss_fn = LPIPS(net_type="alex", reduction="none").to(self.device)

        valid_dataset = self._dataset(dataset, config, mode)
        valid_dataset, chosen = self._maybe_subset(valid_dataset, config, net)

        # Eq. (21) has to be accumulated over the training set before any
        # variance is meaningful; do it here so no separate script is needed.
        self._prepare_gp(dataset, config)

        validLoader = DataLoader(valid_dataset,
                                 config["Training"]["%s_batch_size" % mode],
                                 shuffle=False, drop_last=False)

        MU, SIG, ENS_GHI = [], [], []
        GT, KORG, ICLR = [], [], []
        pixel_sqerr_ps, pixel_abserr_ps, lpips_ps, index_ps = [], [], [], []
        reg_sqerr_ps, reg_abserr_ps = [], []
        pixel_sqerr_sum, pixel_abserr_sum, pixel_elem = 0.0, 0.0, 0
        sample_counter = 0
        samples_list = []
        max_dumps = int(net.get("max_image_dumps", 8))

        pbar = tqdm(validLoader)
        for input_index, data in enumerate(pbar):
            b = self._unpack(data)

            with torch.no_grad():
                # ---- Sec. 2.2: N possible futures -------------------------- #
                gen = self.model.video.sample_sequence(
                    b["x_frames"], n_samples=n_ens, steps=ddim_steps)      # [B,N,T,C,H,W]
                B, _, T, C, H, W = gen.shape

                # ---- Sec. 2.3: one multimodal pass per generated sequence --- #
                feats = self.model.forecaster.resnet.encode_sequence(
                    gen.reshape(B * n_ens, T, C, H, W)).reshape(B, n_ens, T, -1)

                mu_b, sig_b, ens_ghi_b = [], [], []
                for i in range(n_ens):
                    mu, var = self.model.forecaster(
                        None, b["z_x"], b["u_x"], b["u_y"],
                        with_variance=True, img_feats=feats[:, i])
                    # normalised -> W/m^2; the transform is affine given Iclr,
                    # so each component stays Gaussian and the mixture is exact
                    mu_b.append(self.target.decode_mean(mu, b["iclr_y"]).cpu())
                    sig_b.append(self.target.decode_std(var.sqrt(), b["iclr_y"]).cpu())
                    # framing (b): this member's FRAMES through the shared regressor
                    ens_ghi_b.append(self._regress(gen[:, i]).cpu())

                MU.append(torch.stack(mu_b, 1))                            # [B,N,T]
                SIG.append(torch.stack(sig_b, 1))
                ENS_GHI.append(torch.stack(ens_ghi_b, 1))

                # ---- image-quality metrics on the ensemble-mean frames ----- #
                Y_predict = gen.mean(dim=1)                                # [B,T,C,H,W]
                pu8 = (self._to_unit(Y_predict) * 255).to(torch.uint8)
                tu8 = (self._to_unit(b["y_frames"]) * 255).to(torch.uint8)

                reg_ghi = self._regress(Y_predict)                         # [B,T]
                ghi_target = b["ghi_y"]
                reg_sqerr_ps.append(((reg_ghi - ghi_target) ** 2).cpu())
                reg_abserr_ps.append((reg_ghi - ghi_target).abs().cpu())

                pdiff = tu8.to(torch.int16) - pu8.to(torch.int16)
                pixel_sqerr_sum += float((pdiff.float() ** 2).sum())
                pixel_abserr_sum += float(pdiff.abs().sum())
                pixel_elem += pdiff.numel()
                _pd = pdiff.float().reshape(B, T, -1)
                pixel_sqerr_ps.append((_pd ** 2).mean(dim=2).cpu())
                pixel_abserr_ps.append(_pd.abs().mean(dim=2).cpu())
                lpips_ps.append(self._per_sample_lpips(lpips_loss_fn, pu8, tu8).cpu())

                GT.append(ghi_target.cpu())
                ICLR.append(b["iclr_y"].cpu())
                k_i = b["ghi_x"][:, -1] / b["iclr_x"][:, -1].clamp(min=iclr_floor)
                KORG.append(k_i.cpu())

            # Manifest-keyed frame export (aligned, unlike evaluation()'s shuffled
            # dumps). No-op for batches that don't overlap frame_manifest.csv, and a
            # no-op entirely unless that file is present. Uses the ensemble-mean
            # frame (pu8), same representative frame this method already scores.
            # Written into THIS run's own results folder (test-images/{case_id}/),
            # not a shared path -- see models/common/frame_export.py.
            if mode == "test":
                dump_manifest_frames("xiong", sample_counter, tu8, pu8,
                                      out_dir=os.path.join(config["logging_version_path"], "test-images"))

            index_ps.append(torch.arange(sample_counter, sample_counter + B))
            sample_counter += B
            if len(samples_list) < max_dumps:
                samples_list.append({"predictions": pu8[0], "ground_truth": tu8[0]})

            running_mse = pixel_sqerr_sum / pixel_elem
            pbar.set_description(
                "Test[forecast] EP:%d [%d] PSNR:%.2f LPIPS:%.4f"
                % (epoch, input_index,
                   20.0 * math.log10(255.0 / math.sqrt(max(running_mse, 1e-8))),
                   float(torch.cat(lpips_ps).mean())))

        # ------------------------------------------------------------------ #
        MU = torch.cat(MU).numpy().astype(np.float64)          # [N,M,T]
        SIG = torch.cat(SIG).numpy().astype(np.float64)
        ENS_GHI = torch.cat(ENS_GHI).numpy().astype(np.float64)
        GT = torch.cat(GT).numpy().astype(np.float64)          # [N,T]
        ICLR = np.clip(torch.cat(ICLR).numpy().astype(np.float64), iclr_floor, None)
        KORG = torch.cat(KORG).numpy().astype(np.float64)
        SPM = KORG[:, None] * ICLR

        pixel_sqerr_all = torch.cat(pixel_sqerr_ps, dim=0)
        pixel_abserr_all = torch.cat(pixel_abserr_ps, dim=0)
        lpips_all = torch.cat(lpips_ps, dim=0)
        index_all = torch.cat(index_ps, dim=0)
        reg_sqerr_all = torch.cat(reg_sqerr_ps, dim=0)
        reg_abserr_all = torch.cat(reg_abserr_ps, dim=0)

        result = self._forecast_metrics(MU, SIG, ENS_GHI, GT, SPM,
                                        pixel_sqerr_all, pixel_abserr_all, lpips_all,
                                        reg_sqerr_all, reg_abserr_all,
                                        pixel_sqerr_sum, pixel_abserr_sum, pixel_elem)
        result["samples_list"] = samples_list

        if mode == "test":
            npz_file = os.path.join(config["ckpt_path"], "xiong_forecast_stats.npz")
            rng = np.random.default_rng(self.xcfg["seed"])
            S = sample_from_mixture(MU, SIG, int(net.get("n_mixture_draws", 100)), rng=rng)
            np.savez_compressed(
                npz_file,
                # --- Xiong's own probabilistic forecast (framing a) ---------- #
                MU=MU.astype(np.float32), SIGMA=SIG.astype(np.float32),
                S=S.astype(np.float32),                 # so analyze_ensemble.py runs as-is
                mu_d=aggregate_mixture(MU, SIG)[0], sigma_d=aggregate_mixture(MU, SIG)[1],
                # --- frames -> shared regressor (framing b, SkyGPT-comparable) #
                ens_ghi=ENS_GHI.astype(np.float32),
                ghi_target=GT, GT=GT,
                # --- joins / baselines --------------------------------------- #
                k_origin=KORG, iclr_y=ICLR, SPM=SPM.astype(np.float32),
                index=index_all.numpy(),
                subset_indices=(np.asarray(chosen) if chosen is not None
                                else np.arange(GT.shape[0])),
                # --- image quality, same keys SkyGPT writes ------------------ #
                pixel_sqerr=pixel_sqerr_all.numpy(), pixel_abserr=pixel_abserr_all.numpy(),
                lpips=lpips_all.numpy(),
                reg_sqerr=reg_sqerr_all.numpy(), reg_abserr=reg_abserr_all.numpy(),
                pixel_mse=np.float64(pixel_sqerr_sum / pixel_elem),
                checkpoint=str(config["ckpt_path"]),
                arm="xiong", n_ensemble=n_ens,
                target_mode=self.xcfg["target"]["mode"],
                add_observation_noise=bool(self.xcfg["gp"]["add_observation_noise"]))
            print("[Xiong forecast-eval] saved -> %s  ([%d,%d,%d])"
                  % (npz_file, MU.shape[0], MU.shape[1], MU.shape[2]))
            print("[Xiong forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)

        return result

    # ---------------------------------------------------------------- #
    def _maybe_subset(self, valid_dataset, config, net):
        """
        Prefer an explicit index file -- that is the only way to guarantee the
        SAME samples as SkyGPT. Falls back to the identical stratified scheme
        (k_origin tertiles, rng seed 1234) if only a size is given.
        """
        idx_file = net.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print(colored("[Xiong] scoring the %d samples from %s (matched to SkyGPT)"
                          % (len(chosen), path), "cyan"))
            return torch.utils.data.Subset(valid_dataset, chosen), chosen

        subset_size = net.get("subset_size", None)
        if not subset_size or int(subset_size) >= len(valid_dataset):
            return valid_dataset, None

        subset_size = int(subset_size)
        k_list = []
        for _i in range(len(valid_dataset)):
            _d = valid_dataset[_i]
            _ghi_i = float(_d["label_x"][-1, 0])
            _iclr_i = max(float(_d["feature_x"][-1, 3]), 1.0)   # feature_x = [h,m,doy,iclr]
            k_list.append(_ghi_i / _iclr_i)
        k_arr = np.asarray(k_list)
        q = np.quantile(k_arr, [1 / 3, 2 / 3])
        bins = [np.where(k_arr < q[0])[0],
                np.where((k_arr >= q[0]) & (k_arr < q[1]))[0],
                np.where(k_arr >= q[1])[0]]
        rng = np.random.default_rng(1234)
        per = subset_size // 3
        chosen = []
        for bb in bins:
            chosen.extend(rng.choice(bb, size=min(per, len(bb)), replace=False).tolist())
        chosen = sorted(chosen)
        np.save(os.path.join(config["ckpt_path"], "xiong_subset_indices.npy"),
                np.asarray(chosen))
        print("[Xiong] stratified subset: %d samples, k tertiles %.3f/%.3f"
              % (len(chosen), q[0], q[1]))
        return torch.utils.data.Subset(valid_dataset, chosen), chosen

    # ---------------------------------------------------------------- #
    def _prepare_gp(self, dataset, config):
        """
        Eq. (21) precision pass over the training set, then the validation-fitted
        observation-noise term. Cached to ckpt_path so re-running forecast()
        doesn't repeat it.

        Without this the GP covariance is still the identity prior and every
        sigma is meaningless -- so it runs unconditionally rather than being a
        separate script somebody can forget.
        """
        gp = self.model.forecaster.gp
        cache = os.path.join(config["ckpt_path"], "xiong_gp_state.pt")

        if os.path.exists(cache):
            state = torch.load(cache, map_location=self.device, weights_only=False)
            gp.precision.copy_(state["precision"])
            gp.cov.copy_(state["cov"])
            gp.sigma_obs.copy_(state["sigma_obs"])
            gp.finalized.fill_(1.0)
            print(colored("[Xiong] GP posterior loaded from %s (sigma_obs=%.4f)"
                          % (cache, float(gp.sigma_obs)), "cyan"))
            return

        print(colored("[Xiong] accumulating GP precision over the training set "
                      "(Eq. 21) -- one-off, cached afterwards", "cyan"))
        self.model.eval()
        gp.reset_precision()

        train_ds = self._dataset(dataset, config, "train")
        loader = DataLoader(train_ds, config["Training"]["valid_batch_size"],
                            shuffle=False, drop_last=False)
        cap = config["Xiong"]["Network"].get("gp_precision_batches", None)
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, desc="GP precision")):
                if cap is not None and i >= int(cap):
                    break
                b = self._unpack(data)
                h = self.model.forecaster.features(
                    self._degraded_future(b), b["z_x"], b["u_x"], b["u_y"])
                gp.update_precision(h)
        gp.finalize()

        if gp.add_observation_noise:
            res, var = [], []
            valid_ds = self._dataset(dataset, config, "valid")
            vloader = DataLoader(valid_ds, config["Training"]["valid_batch_size"],
                                 shuffle=False, drop_last=False)
            with torch.no_grad():
                for i, data in enumerate(tqdm(vloader, desc="sigma_obs")):
                    if i >= 200:
                        break
                    b = self._unpack(data)
                    mu, v = self.model.forecaster(
                        self._degraded_future(b), b["z_x"], b["u_x"], b["u_y"],
                        with_variance=True)
                    res.append((mu - b["z_y"]).flatten())
                    var.append(v.flatten())
            s = gp.fit_observation_noise(torch.cat(res), torch.cat(var))
            print("[Xiong] sigma_obs = %.4f (normalised units), mean epistemic sd = %.4f"
                  % (s, float(torch.cat(var).mean().sqrt())))
        else:
            print(colored("[Xiong] add_observation_noise=false -> Eq. (20) verbatim "
                          "(epistemic only; expect severe under-dispersion)", "yellow"))

        torch.save(dict(precision=gp.precision.cpu(), cov=gp.cov.cpu(),
                        sigma_obs=gp.sigma_obs.cpu()), cache)
        print("[Xiong] GP posterior cached -> %s" % cache)

    # ---------------------------------------------------------------- #
    def _forecast_metrics(self, MU, SIG, ENS_GHI, GT, SPM,
                          pixel_sqerr_all, pixel_abserr_all, lpips_all,
                          reg_sqerr_all, reg_abserr_all,
                          pixel_sqerr_sum, pixel_abserr_sum, pixel_elem):
        """Both framings, one flat dict for the repo's CSV logger."""
        T = GT.shape[1]
        result = {}

        # ---------- image quality (same keys SkyGPT reports) ---------- #
        pixel_mse = pixel_sqerr_sum / pixel_elem
        result["psnr_per_epoch"] = 20.0 * math.log10(255.0 / math.sqrt(pixel_mse))
        result["rmse_per_epoch"] = math.sqrt(pixel_mse)
        result["mae_per_epoch"] = pixel_abserr_sum / pixel_elem
        result["lpips_per_epoch"] = float(lpips_all.mean())

        rmse_h = torch.sqrt(pixel_sqerr_all.mean(dim=0))
        lpips_h = lpips_all.mean(dim=0)
        for i in range(T):
            result["PSNR_%d" % (i + 1)] = float(20.0 * torch.log10(255.0 / rmse_h[i].clamp(min=1e-8)))
            result["LPIPS_%d" % (i + 1)] = float(lpips_h[i])

        reg_rmse_h = torch.sqrt(reg_sqerr_all.mean(dim=0))
        reg_mae_h = reg_abserr_all.mean(dim=0)
        for i in range(T):
            result["Reg[RMSE]_%d" % (i + 1)] = float(reg_rmse_h[i])
            result["Reg[MAE]_%d" % (i + 1)] = float(reg_mae_h[i])
        result["Reg[RMSE]_avg"] = float(reg_rmse_h.mean())
        result["Reg[MAE]_avg"] = float(reg_mae_h.mean())

        # ---------- (a) end-to-end forecast from the GPA head ---------- #
        mu_d, _ = aggregate_mixture(MU, SIG)
        spm_rmse = pooled_rmse(SPM, GT)
        spm_mae = pooled_mae(SPM, GT)

        # FS and CRPSS divide by the persistence baseline. On real SIRTA data
        # SPM RMSE is ~123 W/m^2, so this never binds -- but if it ever does,
        # the skill scores explode into meaningless magnitudes instead of
        # failing, which is worse than failing. Say so loudly.
        if float(spm_rmse.mean()) < 1.0 or float(spm_mae.mean()) < 1.0:
            print(colored("[Xiong] WARNING: smart-persistence RMSE is ~0 (%.3f). FS/CRPSS "
                          "are meaningless here. On real data expect ~123 W/m^2 -- this "
                          "means k_origin*Iclr reproduces the target exactly, i.e. the "
                          "Iclr column or the split is not what you think."
                          % float(spm_rmse.mean()), "red"))
        spm_rmse = np.clip(spm_rmse, 1e-3, None)
        spm_mae = np.clip(spm_mae, 1e-3, None)

        single = float(np.mean([pooled_rmse(MU[:, j, :], GT).mean() for j in range(MU.shape[1])]))
        rmse_ens = pooled_rmse(mu_d, GT)
        crps = crps_mixture(MU, SIG, GT).mean(axis=0)
        wis, cov = mwis(MU, SIG, GT)

        result["GHI[single-sample RMSE]"] = single
        result["GHI[ensemble RMSE]"] = float(rmse_ens.mean())
        result["GHI[SPM RMSE]"] = float(spm_rmse.mean())
        result["GHI[FS]"] = float((1.0 - rmse_ens / spm_rmse).mean())
        result["GHI[CRPS]"] = float(crps.mean())
        result["GHI[CRPSS]"] = float((1.0 - crps / spm_mae).mean())
        result["GHI[MWIS]"] = float(wis.mean())
        result["GHI[spread]"] = float(SIG.mean())
        for k, v in sorted(cov.items()):
            result["GHI[PICP_%d]" % int(round(100 * k))] = v
        for i in range(T):
            result["GHI[RMSE]_%d" % (i + 1)] = float(rmse_ens[i])
            result["GHI[CRPS]_%d" % (i + 1)] = float(crps[i])

        # ---------- (b) frames -> shared regressor (SkyGPT-comparable) -- #
        reg_mean = ENS_GHI.mean(axis=1)
        reg_rmse_ens = pooled_rmse(reg_mean, GT)
        reg_crps = crps_ensemble_fair(ENS_GHI, GT).mean(axis=0)
        result["RegGHI[ensemble RMSE]"] = float(reg_rmse_ens.mean())
        result["RegGHI[FS]"] = float((1.0 - reg_rmse_ens / spm_rmse).mean())
        result["RegGHI[CRPS]"] = float(reg_crps.mean())
        result["RegGHI[CRPSS]"] = float((1.0 - reg_crps / spm_mae).mean())
        result["RegGHI[spread]"] = float(ENS_GHI.std(axis=1).mean())

        print("\n" + "=" * 72)
        print("XIONG FORECAST  (N=%d, %d samples)" % (MU.shape[1], GT.shape[0]))
        print("  (a) direct GPA head : RMSE %7.1f  FS %+.3f  CRPS %6.1f  CRPSS %+.3f"
              % (result["GHI[ensemble RMSE]"], result["GHI[FS]"],
                 result["GHI[CRPS]"], result["GHI[CRPSS]"]))
        print("  (b) frames->regressor: RMSE %7.1f  FS %+.3f  CRPS %6.1f  CRPSS %+.3f"
              % (result["RegGHI[ensemble RMSE]"], result["RegGHI[FS]"],
                 result["RegGHI[CRPS]"], result["RegGHI[CRPSS]"]))
        print("      smart persistence: RMSE %7.1f" % result["GHI[SPM RMSE]"])
        print("  image: Reg[RMSE] %.1f  LPIPS %.4f  Pixel-RMSE %.2f"
              % (result["Reg[RMSE]_avg"], result["lpips_per_epoch"],
                 result["rmse_per_epoch"]))
        print("  (a) vs (b) on identical samples measures the extraction-pipeline")
        print("      penalty directly -- the caveat the SkyGPT comparison needs.")
        print("=" * 72 + "\n")
        return result
