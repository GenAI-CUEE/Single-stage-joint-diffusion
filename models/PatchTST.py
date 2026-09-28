"""
models/PatchTST.py

PatchTST (Nie et al., ICLR 2023, "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers"), ported from thuml/Time-Series-Library
(TSLib_original/models/PatchTST.py; supporting layers in
models/PatchTST_utils/layers.py) and wrapped in the same Model interface the
rest of this repo uses (see models/Xiong.py's docstring):

    Model(configs)          built by main.py
      .num_param            logged by main.py
      .optimizer            wrapped by train.py's ReduceLROnPlateau
      .model                nn.Module; main.py's generic checkpoint path does
                            module.model.load_state_dict(ckpt['state_dict'])
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats

Group 3 baseline: scalar-only, deterministic point forecaster. No image
*model* -- predicts GHI directly from GHI history (label_x[:,:,0]). "image" is
still passed in attribute_list (SeqDataset_SIRTA processes it unconditionally
regardless of attribute_list -- a dataloader bug, see
this was left as-is rather than
touching a dataloader every other model depends on).

patch_len/stride are NOT TSLib's own defaults (16/8) -- with this repo's much
shorter seq_length=8, those defaults collapse the patching to a single patch
(degenerate transformer over a sequence of length 1). See
configs/2023/PatchTST_SIRTA16x15.yaml for the values actually used here.

include_future_iclr (config toggle) is NOT part of canonical PatchTST -- same
small, zero-initialised additive head over the known-future clear-sky index
as models/DLinear.py, toggled for the with-vs-without-Iclr ablation.
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from models.PatchTST_utils.layers import (AttentionLayer, Encoder, EncoderLayer,
                                          FullAttention, PatchEmbedding)
from models.common.metrics import forecast_skill, pooled_mae, pooled_rmse


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        return x.transpose(*self.dims).contiguous() if self.contiguous else x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):   # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class _PatchTSTCore(nn.Module):
    """TSLib_original/models/PatchTST.py's Model.forecast() path, trimmed to
    a single channel (GHI) -- patch_len/stride tuned for seq_length=8 (see
    module docstring)."""

    def __init__(self, seq_len, pred_len, patch_len, stride, d_model, n_heads,
                 e_layers, d_ff, dropout, factor=5, activation="gelu"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)),
        )

        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc):
        # x_enc: [B, seq_len, 1]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)                  # [B, 1, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)            # [B, 1, d_model, patch_num]

        dec_out = self.head(enc_out)                     # [B, 1, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                # [B, pred_len, 1]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out.squeeze(-1)                        # [B, pred_len]


class PatchTSTNet(nn.Module):
    """PatchTST core + optional additive future-Iclr correction head. One
    nn.Module so EarlyStopping's generic state_dict save/load captures both
    pieces together (see models/DLinear.py's DLinearNet, same pattern)."""

    def __init__(self, seq_len, pred_len, include_future_iclr, **core_kwargs):
        super().__init__()
        self.include_future_iclr = include_future_iclr
        self.core = _PatchTSTCore(seq_len, pred_len, **core_kwargs)
        if include_future_iclr:
            self.iclr_head = nn.Linear(pred_len, pred_len)
            nn.init.zeros_(self.iclr_head.weight)
            nn.init.zeros_(self.iclr_head.bias)

    def forward(self, ghi_hist, future_iclr=None):
        out = self.core(ghi_hist)
        if self.include_future_iclr:
            out = out + self.iclr_head(future_iclr)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        net = configs["PatchTST"]["Network"]
        learn = configs["PatchTST"]["Learning"]

        self.is_train = configs["Training"]["is_train"]
        self.which_model = "PatchTST"

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]
        self.iclr_idx = net.get("iclr_idx", 3)
        self.include_future_iclr = bool(net.get("include_future_iclr", False))

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.model = PatchTSTNet(
            seq_len=self.seq_length, pred_len=self.pred_length,
            include_future_iclr=self.include_future_iclr,
            patch_len=net.get("patch_len", 2), stride=net.get("stride", 1),
            d_model=net.get("d_model", 32), n_heads=net.get("n_heads", 4),
            e_layers=net.get("e_layers", 2), d_ff=net.get("d_ff", 64),
            dropout=net.get("dropout", 0.1), factor=net.get("factor", 5),
            activation=net.get("activation", "gelu"),
        ).to(self.device)

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[PatchTST] Num. params = %d  include_future_iclr=%s"
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
        """Ported from models/Xiong.py::_maybe_subset (see models/DLinear.py's
        copy of the same helper)."""
        idx_file = ev.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print("[PatchTST] scoring the %d samples from %s (matched to SkyGPT)"
                  % (len(chosen), path))
            return Subset(test_dataset, chosen), chosen
        return test_dataset, None

    # ---------------------------------------------------------------- #
    def forecast(self, dataset, config, epoch, mode="test"):
        """The comparison path -- see models/DLinear.py::forecast() for the
        full rationale (identical design, different core network)."""
        net = config["PatchTST"]["Network"]
        ev = config["PatchTST"].get("Eval", {})

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
            S = np.tile(pred_all[:, None, :], (1, M, 1)).astype(np.float32)

            index_all = np.asarray(chosen) if chosen is not None else np.arange(gt_all.shape[0])

            npz_file = os.path.join(config["ckpt_path"], "patchtst_forecast_stats.npz")
            np.savez_compressed(
                npz_file,
                S=S, GT=gt_all.astype(np.float32),
                k_origin=korg_all.astype(np.float32), iclr_y=iclr_all.astype(np.float32),
                SPM=spm.astype(np.float32),
                pred=pred_all.astype(np.float32),
                index=index_all, subset_indices=index_all,
                checkpoint=str(config["ckpt_path"]),
                arm="patchtst", include_future_iclr=self.include_future_iclr)

            print("[PatchTST forecast-eval] saved -> %s  (N=%d, M=%d, H=%d)"
                  % (npz_file, S.shape[0], M, S.shape[2]))
            print("[PatchTST forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)
            print("[PatchTST forecast-eval] FS mean %+.3f  CRPSS mean %+.3f"
                  % (fs.mean(), crpss.mean()))
            print("[PatchTST forecast-eval] RMSE pooled %.2f  FS pooled %+.3f"
                  % (rmse_pooled, fs_pooled))

        return result
