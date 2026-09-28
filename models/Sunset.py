"""
models/Sunset.py

Sunset (Nie et al., Stanford solar forecasting -- SUNSET model), ported from
the third-party TF/Keras reimplementation at
https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset
(Sunset_original/models/SUNSET_forecast.ipynb, untracked reference, same
precedent as TSLib_original/ and SkyGPT_original/) and wrapped in the same
Model interface the rest of this repo uses (see models/DLinear.py's
docstring, which this file mirrors file-for-file):

    Model(configs)          built by main.py
      .num_param            logged by main.py
      .optimizer            wrapped by train.py's ReduceLROnPlateau
      .model                nn.Module; main.py's generic checkpoint path does
                            module.model.load_state_dict(ckpt['state_dict'])
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats

Unlike Group 3 (DLinear/PatchTST/TimeXer -- scalar-only, image tensor loaded
but discarded), Sunset is an **image-based CNN** GHI forecaster: it genuinely
consumes `image_x` (the past seq_length frames), channel-stacked, alongside
the past GHI log.

Deviations from the original paper (deliberate, see DLinear_PatchTST_TimeXer_
task.md Part D for the full rationale):
  - PyTorch, not TF/Keras.
  - Direct multi-horizon output (Dense(pred_length=8) instead of Dense(1) for
    a single 15-min-ahead step) -- one forward pass predicts all 8 horizons,
    matching every other model in this table.
  - Deterministic single model + M-tile ensembling for analyze_ensemble.py
    (see forecast() below), not the original's 10-independently-trained-model
    bagging ensemble -- "similar behaviour to DLinear/PatchTST/TimeXer" per
    the task instruction, and 1x training cost instead of 10x.
  - include_future_iclr ablation toggle added (not in the original paper),
    same zero-initialised additive head pattern as DLinear -- for consistency
    with the rest of the comparison table.

Channel-stacking (the original's `(64,64,48)` = 16 frames x 3 channels input)
comes for free from the shared dataloader: configs/2023/Sunset_SIRTA16x15.yaml sets
Data.time_channel_coupling_enable: True, so SeqDataset_SIRTA itself emits
image_x already reshaped to (B, seq_length*image_channels, H, W) -- no reshape
code needed here (see dataloaders/preprocessing_SIRTA_sequence.py:307-317).
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from losses.Sunset.losses import MSE_Loss
from models.common.metrics import forecast_skill, pooled_mae, pooled_rmse


class _SunsetCore(nn.Module):
    """The CNN + dense regression head, ported from sunset_model() in
    SUNSET_forecast.ipynb (see this file's module docstring for the exact
    Keras architecture it mirrors)."""

    def __init__(self, seq_len, pred_len, image_channels, image_size,
                 num_filters=24, kernel_size=3, pool_size=2,
                 dense_size=1024, drop_rate=0.4):
        super().__init__()
        in_ch = seq_len * image_channels

        self.conv1 = nn.Conv2d(in_ch, num_filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(pool_size, stride=2)

        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.pool2 = nn.MaxPool2d(pool_size, stride=2)

        # Two stride-2 pools halve the spatial size twice: image_size -> //4.
        feat_side = image_size // 4
        flat_dim = num_filters * 2 * feat_side * feat_side

        self.fc1 = nn.Linear(flat_dim + seq_len, dense_size)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(dense_size, dense_size)
        self.drop2 = nn.Dropout(drop_rate)
        self.fc_out = nn.Linear(dense_size, pred_len)

    def forward(self, image_stack, ghi_hist):
        # image_stack: [B, seq_len*image_channels, H, W] ; ghi_hist: [B, seq_len]
        x = torch.relu(self.bn1(self.conv1(image_stack)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.flatten(1)
        x = torch.cat([x, ghi_hist], dim=1)
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return self.fc_out(x)   # [B, pred_len]


class SunsetNet(nn.Module):
    """Sunset core + optional additive future-Iclr correction head. One
    nn.Module so EarlyStopping's generic state_dict save/load captures both
    pieces together (same pattern as models/DLinear.py's DLinearNet)."""

    def __init__(self, seq_len, pred_len, image_channels, image_size,
                 num_filters, kernel_size, pool_size, dense_size, drop_rate,
                 include_future_iclr):
        super().__init__()
        self.include_future_iclr = include_future_iclr
        self.core = _SunsetCore(seq_len, pred_len, image_channels, image_size,
                                 num_filters, kernel_size, pool_size,
                                 dense_size, drop_rate)
        if include_future_iclr:
            self.iclr_head = nn.Linear(pred_len, pred_len)
            nn.init.zeros_(self.iclr_head.weight)
            nn.init.zeros_(self.iclr_head.bias)

    def forward(self, image_stack, ghi_hist, future_iclr=None):
        out = self.core(image_stack, ghi_hist)   # [B, pred_len]
        if self.include_future_iclr:
            out = out + self.iclr_head(future_iclr)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        net = configs["Sunset"]["Network"]
        learn = configs["Sunset"]["Learning"]

        self.is_train = configs["Training"]["is_train"]
        self.which_model = "Sunset"

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]
        self.image_channels = configs["ImageSequence"]["image_channels"]
        self.image_size = configs["ImageSequence"]["image_size"]
        self.iclr_idx = net.get("iclr_idx", 3)
        self.include_future_iclr = bool(net.get("include_future_iclr", False))

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.num_workers = 0
        self.pin_memory = (self.device.type == "cuda")
        self.multiprocessing_context = None

        self.model = SunsetNet(
            seq_len=self.seq_length, pred_len=self.pred_length,
            image_channels=self.image_channels, image_size=self.image_size,
            num_filters=net.get("num_filters", 24),
            kernel_size=net.get("kernel_size", 3),
            pool_size=net.get("pool_size", 2),
            dense_size=net.get("dense_size", 1024),
            drop_rate=net.get("drop_rate", 0.4),
            include_future_iclr=self.include_future_iclr,
        ).to(self.device)

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[Sunset] Num. params = %d  include_future_iclr=%s"
              % (self.num_param, self.include_future_iclr))

        self.loss_fn = MSE_Loss()

        if self.is_train:
            self.subsampling_enable = configs["Training"]["subsampling_enable"]
            self.subsampling_rate = configs["Training"]["subsampling_rate"]
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learn["learning_rate"],
                betas=tuple(learn.get("learning_betas", [0.9, 0.999])))

    # ---------------------------------------------------------------- #
    def _batch_tensors(self, data):
        image_stack = (data["image_x"].float().to(self.device) / 255.0) - 0.5  # [B, seq_len*C, H, W]
        ghi_hist = data["label_x"][:, :, 0].float().to(self.device)            # [B, seq_len]
        ghi_target = data["label_y"][:, :, 0].float().to(self.device)          # [B, pred_len]
        future_iclr = (data["feature_y"][:, :, self.iclr_idx].float().to(self.device)
                       if self.include_future_iclr else None)
        return image_stack, ghi_hist, ghi_target, future_iclr

    # ---------------------------------------------------------------- #
    def train(self, dataset, config, epoch):
        full_train_dataset = dataset(config, self.device, mode="train",
                                     attribute_list=["label", "feature", "image"])

        if self.subsampling_enable:
            subset_size = int(self.subsampling_rate * len(full_train_dataset))
            indices = torch.randperm(len(full_train_dataset))[:subset_size].tolist()
            sampler = SubsetRandomSampler(indices)
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     sampler=sampler, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, pin_memory=self.pin_memory,
                                     multiprocessing_context=self.multiprocessing_context)
        else:
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     shuffle=True, drop_last=True,
                                     num_workers=self.num_workers, pin_memory=self.pin_memory,
                                     multiprocessing_context=self.multiprocessing_context)

        self.model.train()

        training_loss = []
        sqerr_sum = 0.0
        elem_count = 0

        pbar = tqdm(trainLoader)
        for input_index, data in enumerate(pbar):
            image_stack, ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            self.optimizer.zero_grad()
            pred = self.model(image_stack, ghi_hist, future_iclr)
            loss = self.loss_fn(pred, ghi_target)
            loss.backward()
            self.optimizer.step()

            training_loss.append(loss.item())
            sqerr_sum += ((pred.detach() - ghi_target) ** 2).sum().item()
            elem_count += ghi_target.numel()

            pbar.set_description(
                "Train EP:%d [%d] Loss:%.4f RMSE:%.2f"
                % (epoch, input_index, sum(training_loss) / len(training_loss),
                   math.sqrt(sqerr_sum / elem_count)))

        return {
            "loss_per_epoch": sum(training_loss) / len(training_loss),
            "psnr_per_epoch": math.sqrt(sqerr_sum / elem_count),
        }

    # ---------------------------------------------------------------- #
    def evaluation(self, dataset, config, epoch, mode="valid"):
        self.model.eval()

        valid_dataset = dataset(config, self.device, mode=mode,
                                attribute_list=["label", "feature", "image"])
        validLoader = DataLoader(valid_dataset, config["Training"]["%s_batch_size" % mode],
                                 shuffle=False, drop_last=False,
                                 num_workers=self.num_workers, pin_memory=self.pin_memory,
                                 multiprocessing_context=self.multiprocessing_context)

        valid_loss = []
        sqerr_sum = 0.0
        elem_count = 0

        pbar = tqdm(validLoader)
        for input_index, data in enumerate(pbar):
            image_stack, ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            with torch.no_grad():
                pred = self.model(image_stack, ghi_hist, future_iclr)
                loss = self.loss_fn(pred, ghi_target)

            valid_loss.append(loss.item())
            sqerr_sum += ((pred - ghi_target) ** 2).sum().item()
            elem_count += ghi_target.numel()

            tag = "Valid" if mode == "valid" else "Test"
            pbar.set_description(
                "%s EP:%d [%d] Loss:%.4f RMSE:%.2f"
                % (tag, epoch, input_index, sum(valid_loss) / len(valid_loss),
                   math.sqrt(sqerr_sum / elem_count)))

        return {
            "loss_per_epoch": sum(valid_loss) / len(valid_loss),
            "psnr_per_epoch": math.sqrt(sqerr_sum / elem_count),
            "samples_list": [],
        }

    # ---------------------------------------------------------------- #
    def _maybe_subset(self, test_dataset, config, ev):
        """Ported verbatim from models/DLinear.py -- prefer an explicit index
        file, which is the only way to guarantee the SAME samples as
        SkyGPT/Xiong/Group 3."""
        idx_file = ev.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print("[Sunset] scoring the %d samples from %s (matched to SkyGPT)"
                  % (len(chosen), path))
            return Subset(test_dataset, chosen), chosen
        return test_dataset, None

    # ---------------------------------------------------------------- #
    def forecast(self, dataset, config, epoch, mode="test"):
        """The comparison path -- targets the genuinely-future label_y, like
        SkyGPT's/Xiong's/DLinear's forecast(). Reports pooled per-horizon
        RMSE/MAE, FS vs. Smart Persistence (SPM), and CRPS := MAE (the exact
        identity for a deterministic forecast -- see
        the tradeoff discussion elsewhere in this file).
        """
        net = config["Sunset"]["Network"]
        ev = config["Sunset"].get("Eval", {})

        self.model.eval()

        test_dataset_full = dataset(config, self.device, mode=mode,
                                    attribute_list=["label", "feature", "image"])
        test_dataset, chosen = self._maybe_subset(test_dataset_full, config, ev)

        testLoader = DataLoader(test_dataset, config["Training"]["test_batch_size"],
                                shuffle=False, drop_last=False,
                                num_workers=self.num_workers, pin_memory=self.pin_memory,
                                multiprocessing_context=self.multiprocessing_context)

        preds, gts, korgs, iclrs = [], [], [], []

        pbar = tqdm(testLoader)
        for input_index, data in enumerate(pbar):
            image_stack, ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            ghi_i = data["label_x"][:, -1, 0].float()
            iclr_i = data["feature_x"][:, -1, self.iclr_idx].float().clamp(min=1.0)
            iclr_y = data["feature_y"][:, :, self.iclr_idx].float().clamp(min=1.0)

            with torch.no_grad():
                pred = self.model(image_stack, ghi_hist, future_iclr).cpu()

            preds.append(pred)
            gts.append(ghi_target.cpu())
            korgs.append(ghi_i / iclr_i)
            iclrs.append(iclr_y)

            pbar.set_description("Test[forecast] EP:%d [%d]" % (epoch, input_index))

        pred_all = torch.cat(preds, dim=0).numpy().astype(np.float64)    # [N,H]
        gt_all = torch.cat(gts, dim=0).numpy().astype(np.float64)        # [N,H]
        korg_all = torch.cat(korgs, dim=0).numpy().astype(np.float64)    # [N]
        iclr_all = torch.cat(iclrs, dim=0).numpy().astype(np.float64)    # [N,H]

        spm = korg_all[:, None] * iclr_all
        rmse = pooled_rmse(pred_all, gt_all)
        mae = pooled_mae(pred_all, gt_all)
        spm_rmse = pooled_rmse(spm, gt_all)
        spm_mae = pooled_mae(spm, gt_all)
        fs = forecast_skill(rmse, spm_rmse)
        crps = mae.copy()                      # deterministic: CRPS == MAE
        crpss = 1.0 - crps / spm_mae

        # Headline metric: RMSE pooled over
        # HORIZONS too, not just samples -- sqrt(mean(RMSE_h^2)), NOT
        # rmse.mean(). rmse/spm_rmse above are already correctly pooled over
        # samples per horizon; this is the separate horizon-axis collapse.
        rmse_pooled = float(np.sqrt(np.mean(rmse ** 2)))
        spm_rmse_pooled = float(np.sqrt(np.mean(spm_rmse ** 2)))
        fs_pooled = 1.0 - rmse_pooled / spm_rmse_pooled

        result = {
            "loss_per_epoch": float(np.mean((pred_all - gt_all) ** 2)),
            "psnr_per_epoch": float(rmse.mean()),
            "samples_list": [],
            "rmse_mean": float(rmse.mean()), "mae_mean": float(mae.mean()),
            "fs_mean": float(fs.mean()), "crps_mean": float(crps.mean()),
            "crpss_mean": float(crpss.mean()),
            "rmse_pooled": rmse_pooled, "fs_pooled": fs_pooled,
        }
        for h in range(rmse.shape[0]):
            result["RMSE_%d" % (h + 1)] = float(rmse[h])
            result["MAE_%d" % (h + 1)] = float(mae[h])
            result["FS_%d" % (h + 1)] = float(fs[h])
            result["CRPSS_%d" % (h + 1)] = float(crpss[h])

        if mode == "test":
            M = int(ev.get("n_ensemble_tile", 100))
            # analyze_ensemble.py requires a genuine [N,M,H] ensemble array --
            # M identical copies gives exactly zero spread; its own CRPS
            # estimator then algebraically collapses to exactly MAE. Same
            # deterministic tile trick as models/DLinear.py -- see this
            # file's module docstring for why (not the paper's 10-model
            # bagging ensemble).
            S = np.tile(pred_all[:, None, :], (1, M, 1)).astype(np.float32)

            index_all = np.asarray(chosen) if chosen is not None else np.arange(gt_all.shape[0])

            npz_file = os.path.join(config["ckpt_path"], "sunset_forecast_stats.npz")
            np.savez_compressed(
                npz_file,
                S=S, GT=gt_all.astype(np.float32),
                k_origin=korg_all.astype(np.float32), iclr_y=iclr_all.astype(np.float32),
                SPM=spm.astype(np.float32),
                pred=pred_all.astype(np.float32),
                index=index_all, subset_indices=index_all,
                checkpoint=str(config["ckpt_path"]),
                arm="sunset", include_future_iclr=self.include_future_iclr)

            print("[Sunset forecast-eval] saved -> %s  (N=%d, M=%d, H=%d)"
                  % (npz_file, S.shape[0], M, S.shape[2]))
            print("[Sunset forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)
            print("[Sunset forecast-eval] FS mean %+.3f  CRPSS mean %+.3f"
                  % (fs.mean(), crpss.mean()))
            print("[Sunset forecast-eval] RMSE pooled %.2f  FS pooled %+.3f"
                  % (rmse_pooled, fs_pooled))

        return result
