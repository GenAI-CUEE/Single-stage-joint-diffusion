"""
models/DLinear.py

DLinear (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series
Forecasting?"), ported from thuml/Time-Series-Library
(TSLib_original/models/DLinear.py, see models/DLinear_utils/layers.py for the
ported series_decomp/moving_avg blocks) and wrapped in the same Model
interface the rest of this repo uses (see models/Xiong.py's docstring):

    Model(configs)          built by main.py
      .num_param            logged by main.py
      .optimizer            wrapped by train.py's ReduceLROnPlateau
      .model                nn.Module; main.py's generic checkpoint path does
                            module.model.load_state_dict(ckpt['state_dict'])
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats

Group 3 baseline: scalar-only, deterministic point forecaster. No image
*model* -- predicts GHI directly from GHI history (label_x[:,:,0]), only ever
reading SeqDataset_SIRTA's "label"/"feature" arrays. "image" is still passed
in attribute_list and does load the 3.3GB SIRTA image pickle: SeqDataset_SIRTA
processes "image" unconditionally regardless of attribute_list (a real bug --
only the pickle *merge* is actually gated), and the fix was deferred rather
than touching a dataloader every other model depends on.

include_future_iclr (config toggle, see configs/2023/DLinear_SIRTA16x15.yaml) is
NOT part of canonical DLinear -- it's a small, zero-initialised additive head
over the known-future clear-sky index (feature_y[:,:,iclr_idx]), the same
free signal Smart Persistence itself exploits. Off by default (canonical
historical-only DLinear); toggled on for the with-vs-without-Iclr ablation.
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from models.DLinear_utils.layers import series_decomp
from models.common.metrics import forecast_skill, pooled_mae, pooled_rmse


class _DLinearCore(nn.Module):
    """The actual DLinear architecture, ported verbatim (single-channel: GHI)."""

    def __init__(self, seq_len, pred_len, moving_avg, individual=False, channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = channels
        self.decompsition = series_decomp(moving_avg)

        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)])
            for i in range(channels):
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / seq_len) * torch.ones([pred_len, seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / seq_len) * torch.ones([pred_len, seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / seq_len) * torch.ones([pred_len, seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, x):
        # x: [B, seq_len, channels]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0), self.channels, self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros(
                trend_init.size(0), self.channels, self.pred_len,
                dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output   # [B, channels, pred_len]
        return x.permute(0, 2, 1)             # [B, pred_len, channels]


class DLinearNet(nn.Module):
    """DLinear core + optional additive future-Iclr correction head. One
    nn.Module so EarlyStopping's generic state_dict save/load captures both
    pieces together."""

    def __init__(self, seq_len, pred_len, moving_avg, individual, include_future_iclr):
        super().__init__()
        self.include_future_iclr = include_future_iclr
        self.core = _DLinearCore(seq_len, pred_len, moving_avg, individual, channels=1)
        if include_future_iclr:
            self.iclr_head = nn.Linear(pred_len, pred_len)
            nn.init.zeros_(self.iclr_head.weight)
            nn.init.zeros_(self.iclr_head.bias)

    def forward(self, ghi_hist, future_iclr=None):
        # ghi_hist: [B, seq_len, 1] ; future_iclr: [B, pred_len] or None
        out = self.core(ghi_hist).squeeze(-1)   # [B, pred_len]
        if self.include_future_iclr:
            out = out + self.iclr_head(future_iclr)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        net = configs["DLinear"]["Network"]
        learn = configs["DLinear"]["Learning"]

        self.is_train = configs["Training"]["is_train"]
        self.which_model = "DLinear"

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]
        self.iclr_idx = net.get("iclr_idx", 3)
        self.include_future_iclr = bool(net.get("include_future_iclr", False))

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.model = DLinearNet(
            seq_len=self.seq_length, pred_len=self.pred_length,
            moving_avg=net.get("moving_avg", 3),
            individual=bool(net.get("individual", False)),
            include_future_iclr=self.include_future_iclr,
        ).to(self.device)

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[DLinear] Num. params = %d  include_future_iclr=%s"
              % (self.num_param, self.include_future_iclr))

        if self.is_train:
            self.subsampling_enable = configs["Training"]["subsampling_enable"]
            self.subsampling_rate = configs["Training"]["subsampling_rate"]
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learn["learning_rate"],
                betas=tuple(learn.get("learning_betas", [0.9, 0.999])))

    # ---------------------------------------------------------------- #
    def _batch_tensors(self, data):
        ghi_hist = data["label_x"][:, :, 0:1].float().to(self.device)      # [B, seq_len, 1]
        ghi_target = data["label_y"][:, :, 0].float().to(self.device)      # [B, pred_len]
        future_iclr = (data["feature_y"][:, :, self.iclr_idx].float().to(self.device)
                       if self.include_future_iclr else None)
        return ghi_hist, ghi_target, future_iclr

    # ---------------------------------------------------------------- #
    def train(self, dataset, config, epoch):
        full_train_dataset = dataset(config, self.device, mode="train",
                                     attribute_list=["label", "feature", "image"])

        if self.subsampling_enable:
            subset_size = int(self.subsampling_rate * len(full_train_dataset))
            indices = torch.randperm(len(full_train_dataset))[:subset_size].tolist()
            sampler = SubsetRandomSampler(indices)
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     sampler=sampler, shuffle=False, drop_last=True)
        else:
            trainLoader = DataLoader(full_train_dataset, config["Training"]["batch_size"],
                                     shuffle=True, drop_last=True)

        self.model.train()

        training_loss = []
        sqerr_sum = 0.0
        elem_count = 0

        pbar = tqdm(trainLoader)
        for input_index, data in enumerate(pbar):
            ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            self.optimizer.zero_grad()
            pred = self.model(ghi_hist, future_iclr)
            loss = torch.nn.functional.mse_loss(pred, ghi_target)
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
                                 shuffle=False, drop_last=False)

        valid_loss = []
        sqerr_sum = 0.0
        elem_count = 0

        pbar = tqdm(validLoader)
        for input_index, data in enumerate(pbar):
            ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            with torch.no_grad():
                pred = self.model(ghi_hist, future_iclr)
                loss = torch.nn.functional.mse_loss(pred, ghi_target)

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
        """Ported from models/Xiong.py::_maybe_subset -- prefer an explicit
        index file, which is the only way to guarantee the SAME samples as
        SkyGPT/Xiong."""
        idx_file = ev.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print("[DLinear] scoring the %d samples from %s (matched to SkyGPT)"
                  % (len(chosen), path))
            return Subset(test_dataset, chosen), chosen
        return test_dataset, None

    # ---------------------------------------------------------------- #
    def forecast(self, dataset, config, epoch, mode="test"):
        """The comparison path -- targets the genuinely-future label_y, like
        SkyGPT's/Xiong's forecast(). Reports pooled per-horizon RMSE/MAE, FS
        vs. Smart Persistence (SPM), and CRPS := MAE (the exact identity for
        a deterministic forecast).
        """
        net = config["DLinear"]["Network"]
        ev = config["DLinear"].get("Eval", {})

        self.model.eval()

        test_dataset_full = dataset(config, self.device, mode=mode,
                                    attribute_list=["label", "feature", "image"])
        test_dataset, chosen = self._maybe_subset(test_dataset_full, config, ev)

        testLoader = DataLoader(test_dataset, config["Training"]["test_batch_size"],
                                shuffle=False, drop_last=False)

        preds, gts, korgs, iclrs = [], [], [], []

        pbar = tqdm(testLoader)
        for input_index, data in enumerate(pbar):
            ghi_hist, ghi_target, future_iclr = self._batch_tensors(data)

            ghi_i = data["label_x"][:, -1, 0].float()
            iclr_i = data["feature_x"][:, -1, self.iclr_idx].float().clamp(min=1.0)
            iclr_y = data["feature_y"][:, :, self.iclr_idx].float().clamp(min=1.0)

            with torch.no_grad():
                pred = self.model(ghi_hist, future_iclr).cpu()

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
            # estimator then algebraically collapses to exactly MAE.
            S = np.tile(pred_all[:, None, :], (1, M, 1)).astype(np.float32)

            index_all = np.asarray(chosen) if chosen is not None else np.arange(gt_all.shape[0])

            npz_file = os.path.join(config["ckpt_path"], "dlinear_forecast_stats.npz")
            np.savez_compressed(
                npz_file,
                S=S, GT=gt_all.astype(np.float32),
                k_origin=korg_all.astype(np.float32), iclr_y=iclr_all.astype(np.float32),
                SPM=spm.astype(np.float32),
                pred=pred_all.astype(np.float32),
                index=index_all, subset_indices=index_all,
                checkpoint=str(config["ckpt_path"]),
                arm="dlinear", include_future_iclr=self.include_future_iclr)

            print("[DLinear forecast-eval] saved -> %s  (N=%d, M=%d, H=%d)"
                  % (npz_file, S.shape[0], M, S.shape[2]))
            print("[DLinear forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)
            print("[DLinear forecast-eval] FS mean %+.3f  CRPSS mean %+.3f"
                  % (fs.mean(), crpss.mean()))
            print("[DLinear forecast-eval] RMSE pooled %.2f  FS pooled %+.3f"
                  % (rmse_pooled, fs_pooled))

        return result
