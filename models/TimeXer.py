"""
models/TimeXer.py

TimeXer (Wang et al., NeurIPS 2024, "Empowering Transformers for Time Series
Forecasting with Exogenous Variables"), ported from thuml/Time-Series-Library
(TSLib_original/models/TimeXer.py; supporting layers in
models/TimeXer_utils/layers.py) and wrapped in the same Model interface the
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
this was left as-is).

include_future_iclr HERE IS DESIGNED DIFFERENTLY FROM DLinear/PatchTST:
those two bolt on a small additive head over future Iclr, since neither
architecture has a native place for a covariate. TimeXer's whole point is its
exogenous cross-attention pathway (EnEmbedding's learned "global token"
cross-attends to a DataEmbedding_inverted-encoded exogenous variable), so the
toggle here routes the SAME real future Iclr sequence
(feature_y[:,:,iclr_idx], length pred_len) through THAT native mechanism
instead. This repo's seq_length == pred_length == 8 makes the exogenous
embedding's expected input length line up cleanly; if that ever changes, the
exogenous embedding's c_in (currently set to pred_length) needs revisiting.

When the toggle is off, the exogenous channel is fed zeros rather than being
architecturally removed -- so the model (and parameter count) is IDENTICAL
either way, and only the data changes, isolating the effect of the signal
itself (unlike DLinear/PatchTST, where the toggle also adds/removes the
additive head's parameters).
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from models.TimeXer_utils.layers import (AttentionLayer, DataEmbedding_inverted,
                                         EnEmbedding, Encoder, EncoderLayer,
                                         FlattenHead, FullAttention)
from models.common.metrics import forecast_skill, pooled_mae, pooled_rmse


class _TimeXerCore(nn.Module):
    """TSLib_original/models/TimeXer.py's Model.forecast() path ("MS" mode:
    one endogenous variable, exogenous variables via cross-attention),
    adapted so the exogenous input is the FUTURE covariate (length pred_len)
    rather than a historical one (see module docstring)."""

    def __init__(self, seq_len, pred_len, patch_len, d_model, n_heads, e_layers,
                 d_ff, dropout, factor=5, activation="gelu", use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        n_vars = 1   # endogenous: GHI only

        self.en_embedding = EnEmbedding(n_vars, d_model, patch_len, dropout)
        # c_in = pred_len: the exogenous input is the FUTURE covariate, whose
        # natural length is pred_len, not seq_len (see module docstring).
        self.ex_embedding = DataEmbedding_inverted(pred_len, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(n_vars, self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, ghi_hist, exo):
        # ghi_hist: [B, seq_len, 1] ; exo: [B, pred_len, 1] (zeros or real future Iclr)
        if self.use_norm:
            means = ghi_hist.mean(1, keepdim=True).detach()
            x = ghi_hist - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = ghi_hist

        en_embed, n_vars = self.en_embedding(x.permute(0, 2, 1))   # [B,1,seq_len] -> patches+glb
        ex_embed = self.ex_embedding(exo)                           # [B, 1, d_model]

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)          # [B, 1, d_model, patch_num+1]

        dec_out = self.head(enc_out)                    # [B, 1, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                # [B, pred_len, 1]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out.squeeze(-1)   # [B, pred_len]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        net = configs["TimeXer"]["Network"]
        learn = configs["TimeXer"]["Learning"]

        self.is_train = configs["Training"]["is_train"]
        self.which_model = "TimeXer"

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]
        self.iclr_idx = net.get("iclr_idx", 3)
        self.include_future_iclr = bool(net.get("include_future_iclr", False))

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        self.model = _TimeXerCore(
            seq_len=self.seq_length, pred_len=self.pred_length,
            patch_len=net.get("patch_len", 2), d_model=net.get("d_model", 32),
            n_heads=net.get("n_heads", 4), e_layers=net.get("e_layers", 2),
            d_ff=net.get("d_ff", 64), dropout=net.get("dropout", 0.1),
            factor=net.get("factor", 5), activation=net.get("activation", "gelu"),
            use_norm=bool(net.get("use_norm", True)),
        ).to(self.device)

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[TimeXer] Num. params = %d  include_future_iclr=%s"
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
        if self.include_future_iclr:
            exo = data["feature_y"][:, :, self.iclr_idx:self.iclr_idx + 1].float().to(self.device)
        else:
            exo = torch.zeros(ghi_target.shape[0], self.pred_length, 1,
                              dtype=torch.float32, device=self.device)
        return ghi_hist, ghi_target, exo

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
            ghi_hist, ghi_target, exo = self._batch_tensors(data)

            self.optimizer.zero_grad()
            pred = self.model(ghi_hist, exo)
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
            ghi_hist, ghi_target, exo = self._batch_tensors(data)

            with torch.no_grad():
                pred = self.model(ghi_hist, exo)
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
            print("[TimeXer] scoring the %d samples from %s (matched to SkyGPT)"
                  % (len(chosen), path))
            return Subset(test_dataset, chosen), chosen
        return test_dataset, None

    # ---------------------------------------------------------------- #
    def forecast(self, dataset, config, epoch, mode="test"):
        """The comparison path -- see models/DLinear.py::forecast() for the
        full rationale (identical design, different core network)."""
        net = config["TimeXer"]["Network"]
        ev = config["TimeXer"].get("Eval", {})

        self.model.eval()

        test_dataset_full = dataset(config, self.device, mode=mode,
                                    attribute_list=["label", "feature", "image"])
        test_dataset, chosen = self._maybe_subset(test_dataset_full, config, ev)

        testLoader = DataLoader(test_dataset, config["Training"]["test_batch_size"],
                                shuffle=False, drop_last=False)

        preds, gts, korgs, iclrs = [], [], [], []

        pbar = tqdm(testLoader)
        for input_index, data in enumerate(pbar):
            ghi_hist, ghi_target, exo = self._batch_tensors(data)

            ghi_i = data["label_x"][:, -1, 0].float()
            iclr_i = data["feature_x"][:, -1, self.iclr_idx].float().clamp(min=1.0)
            iclr_y = data["feature_y"][:, :, self.iclr_idx].float().clamp(min=1.0)

            with torch.no_grad():
                pred = self.model(ghi_hist, exo).cpu()

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

            npz_file = os.path.join(config["ckpt_path"], "timexer_forecast_stats.npz")
            np.savez_compressed(
                npz_file,
                S=S, GT=gt_all.astype(np.float32),
                k_origin=korg_all.astype(np.float32), iclr_y=iclr_all.astype(np.float32),
                SPM=spm.astype(np.float32),
                pred=pred_all.astype(np.float32),
                index=index_all, subset_indices=index_all,
                checkpoint=str(config["ckpt_path"]),
                arm="timexer", include_future_iclr=self.include_future_iclr)

            print("[TimeXer forecast-eval] saved -> %s  (N=%d, M=%d, H=%d)"
                  % (npz_file, S.shape[0], M, S.shape[2]))
            print("[TimeXer forecast-eval] cross-check: python analyze_ensemble.py %s" % npz_file)
            print("[TimeXer forecast-eval] FS mean %+.3f  CRPSS mean %+.3f"
                  % (fs.mean(), crpss.mean()))
            print("[TimeXer forecast-eval] RMSE pooled %.2f  FS pooled %+.3f"
                  % (rmse_pooled, fs_pooled))

        return result
