"""
xiong_multimodal.py

Stages 2 and 3 of the Xiong et al. baseline (paper Sec. 2.3).

  Stage 2  SpectralResNet + regression head
           Trained single-image -> irradiance with MSE, on frames degraded by
           Gaussian noise + blur so the encoder matches the CDM's output
           distribution (Sec. 2.3.3). Afterwards the head is dropped and the
           trunk is FROZEN and reused as the image backbone.

  Stage 3  MultimodalTransformer + GPAHead
           Informer-style encoder/decoder over EARLY-CONCATENATED image and
           power tokens (Sec. 2.3.1, Fig. 4, Eqs. 10-17), with a Gaussian
           process approximation output layer (Sec. 2.3.2, Eqs. 18-21) giving
           (mu_m, sigma_m) per horizon step.

Distance awareness (criterion 1) comes from the random-Fourier-feature GP head;
distance preservation (criterion 2) from the bounded spectral normalisation of
Eq. (22), applied via xiong_common.apply_soft_sn.

DEVIATION FROM THE PAPER, DELIBERATE AND DOCUMENTED
---------------------------------------------------
Eq. (20) gives sigma_m = Phi^T Sigma Phi, which is *epistemic only*. Used
verbatim for regression it produces intervals far too narrow, which would make
the baseline's CRPS/coverage look artificially terrible. GPAHead therefore
carries an optional observation-noise term sigma_obs^2, fitted once on the
validation residuals (`fit_observation_noise`). Keep it on for a fair
comparison; set model.gp.add_observation_noise=false to reproduce Eq. (20)
literally. Report which one you used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.frame_utils import apply_soft_sn, count_params


# =========================================================================== #
# Stage 2: image encoder
# =========================================================================== #
class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_ch))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt, inplace=True)


class SpectralResNet(nn.Module):
    """
    ResNet-18-ish trunk sized for 64x64 sky images, with bounded spectral norm
    (Eq. 22) on every conv/linear inside the residual stack.

    forward(x)          -> pooled feature [B, feat_dim]
    forward_regress(x)  -> scalar prediction [B] (stage-2 pretraining only)
    """

    def __init__(self, in_ch=3, base=64, layers=(2, 2, 2, 2), feat_dim=512,
                 spectral_c=6.0, use_spectral_norm=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        chans = [base, base * 2, base * 4, base * 8]
        blocks, cur = [], base
        for i, (c, n) in enumerate(zip(chans, layers)):
            for j in range(n):
                blocks.append(_BasicBlock(cur, c, stride=2 if (j == 0 and i > 0) else 1))
                cur = c
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Identity() if cur == feat_dim else nn.Linear(cur, feat_dim)
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim, 1)

        if use_spectral_norm:
            apply_soft_sn(self.blocks, c=spectral_c)
            if isinstance(self.proj, nn.Linear):
                apply_soft_sn(self.proj, c=spectral_c)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.proj(h)

    def forward_regress(self, x):
        return self.head(self.forward(x)).squeeze(-1)

    def encode_sequence(self, clip):
        """clip [B,T,3,H,W] -> [B,T,feat_dim] (frames encoded independently)."""
        B, T = clip.shape[:2]
        f = self.forward(clip.reshape(B * T, *clip.shape[2:]))
        return f.reshape(B, T, -1)


# =========================================================================== #
# Stage 3: embeddings (Fig. 4)
# =========================================================================== #
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(nn.Module):
    """Value embedding: 1-D conv, kernel 3 (paper Sec. 2.3.1)."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                              padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):                      # [B,L,c_in]
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """Value + positional (+ optional temporal) embedding, summed."""

    def __init__(self, c_in, d_model, n_temporal=0, dropout=0.1):
        super().__init__()
        self.value = TokenEmbedding(c_in, d_model)
        self.pos = PositionalEmbedding(d_model)
        self.temporal = nn.Linear(n_temporal, d_model) if n_temporal > 0 else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, u=None):
        h = self.value(x) + self.pos(x)
        if self.temporal is not None and u is not None:
            h = h + self.temporal(u)
        return self.drop(h)


# =========================================================================== #
# Stage 3: transformer
# =========================================================================== #
class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln2(x)))


class _DistilLayer(nn.Module):
    """Subsample layer: halves the token axis with stride-2 max pooling."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        h = self.pool(F.elu(self.norm(self.conv(x.transpose(1, 2)))))
        return h.transpose(1, 2)


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model), nn.LayerNorm(d_model),
                                        nn.LayerNorm(d_model))
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, attn_mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=attn_mask,
                                         need_weights=False)[0])
        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, mem, mem, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln3(x)))


class MultimodalTransformer(nn.Module):
    """
    Encoder input : concat([image tokens, power tokens]) -- EARLY concatenation
                    (the paper's ablation shows this beats post-concatenation).
    Decoder input : [last `label_len` of the power history | zeros(pred_len)]
                    plus temporal features for the whole span (Informer style).
    Output        : h(x) [B, pred_len, d_model], handed to the GPA head.
    """

    def __init__(self, img_feat_dim, d_model=128, n_heads=4, e_layers=2, d_layers=1,
                 d_ff=256, dropout=0.1, n_temporal=3, label_len=4, pred_len=8,
                 spectral_c=6.0, use_spectral_norm=True):
        super().__init__()
        self.label_len, self.pred_len = label_len, pred_len
        self.img_embed = DataEmbedding(img_feat_dim, d_model, n_temporal=n_temporal,
                                       dropout=dropout)
        self.pow_embed = DataEmbedding(1, d_model, n_temporal=n_temporal, dropout=dropout)
        self.dec_embed = DataEmbedding(1, d_model, n_temporal=n_temporal, dropout=dropout)

        self.enc_layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.distil = nn.ModuleList(
            [_DistilLayer(d_model) for _ in range(e_layers - 1)])
        self.enc_norm = nn.LayerNorm(d_model)

        self.dec_layers = nn.ModuleList(
            [_DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_spectral_norm:
            apply_soft_sn(self.enc_layers, c=spectral_c)
            apply_soft_sn(self.dec_layers, c=spectral_c)
            apply_soft_sn(self.out_proj, c=spectral_c)

    def forward(self, img_feats, z_hist, u_hist, u_future):
        """
        img_feats [B, Tout, img_feat_dim]  frozen-ResNet features of the
                                           PREDICTED future frames
        z_hist    [B, Tin]                 normalised irradiance history
        u_hist    [B, Tin, nu]             temporal covariates, history
        u_future  [B, Tout, nu]            temporal covariates, horizon
        -> h(x)   [B, pred_len, d_model]
        """
        x_s = self.img_embed(img_feats, u_future)                    # image modality
        x_p = self.pow_embed(z_hist.unsqueeze(-1), u_hist)           # power modality
        enc = torch.cat([x_s, x_p], dim=1)                           # Eqs. (10)-(13)

        for i, layer in enumerate(self.enc_layers):
            enc = layer(enc)
            if i < len(self.distil):
                enc = self.distil[i](enc)
        mem = self.enc_norm(enc)                                     # X_c

        start = z_hist[:, -self.label_len:].unsqueeze(-1)
        holder = torch.zeros(z_hist.shape[0], self.pred_len, 1,
                             device=z_hist.device, dtype=z_hist.dtype)
        dec_in = torch.cat([start, holder], dim=1)
        u_dec = torch.cat([u_hist[:, -self.label_len:], u_future], dim=1)
        dec = self.dec_embed(dec_in, u_dec)

        L = dec.shape[1]
        mask = torch.triu(torch.ones(L, L, device=dec.device, dtype=torch.bool), 1)
        for layer in self.dec_layers:
            dec = layer(dec, mem, attn_mask=mask)                    # Eqs. (14)-(17)

        h = self.out_proj(self.dec_norm(dec))
        return h[:, -self.pred_len:]                                 # h(x)


# =========================================================================== #
# Stage 3: Gaussian process approximation head (Eqs. 18-21)
# =========================================================================== #
class GPAHead(nn.Module):
    """
    Phi   = sqrt(2/D_L) * cos(-A_L h(x) + b_L)            Eq. (18)
    mu    = Phi^T beta                                     Eq. (19)
    sigma = Phi^T Sigma_t Phi                              Eq. (20)
    Sigma_t^{-1} = Sigma_{t-1}^{-1} + sum_i Phi Phi^T      Eq. (21), Sigma_0 = I

    A_L ~ N(0,1) and b_L ~ U(0,2pi) are FIXED buffers; only beta is trained.
    `kernel_scale` divides the (layer-normed) input, which is equivalent to
    A_L ~ N(0, 1/kernel_scale) -- without it the RFF kernel bandwidth is far too
    narrow for a d_model-dimensional input and the head degenerates.
    """

    def __init__(self, d_in, n_rff=1024, kernel_scale=2.0, ridge=1e-3,
                 add_observation_noise=True):
        super().__init__()
        self.n_rff, self.ridge = n_rff, ridge
        self.kernel_scale = kernel_scale
        self.norm = nn.LayerNorm(d_in)
        self.register_buffer("A", torch.randn(n_rff, d_in))
        self.register_buffer("b", torch.rand(n_rff) * 2 * math.pi)
        self.beta = nn.Linear(n_rff, 1, bias=False)
        nn.init.zeros_(self.beta.weight)
        self.register_buffer("precision", torch.eye(n_rff))
        self.register_buffer("cov", torch.eye(n_rff))
        self.register_buffer("sigma_obs", torch.zeros(1))
        # buffer, not a python flag: it has to survive a checkpoint round-trip,
        # otherwise evaluation cannot tell a fitted covariance from the prior
        self.register_buffer("finalized", torch.zeros(1))
        self.add_observation_noise = add_observation_noise

    def features(self, h):
        h = self.norm(h) / math.sqrt(self.kernel_scale)
        return math.sqrt(2.0 / self.n_rff) * torch.cos(-F.linear(h, self.A) + self.b)

    def forward(self, h, with_variance=False):
        phi = self.features(h)                                  # [B,L,D]
        mu = self.beta(phi).squeeze(-1)                          # [B,L]
        if not with_variance:
            return mu
        var = torch.einsum("bld,de,ble->bl", phi, self.cov, phi)
        if self.add_observation_noise:
            var = var + self.sigma_obs ** 2
        return mu, var.clamp(min=1e-8)

    # -- Eq. (21): accumulate the precision over a full pass, no gradients -- #
    @torch.no_grad()
    def reset_precision(self):
        self.precision.copy_(torch.eye(self.n_rff, device=self.precision.device))
        self.finalized.zero_()

    @torch.no_grad()
    def update_precision(self, h):
        phi = self.features(h).reshape(-1, self.n_rff)
        self.precision += phi.t() @ phi

    @torch.no_grad()
    def finalize(self):
        eye = torch.eye(self.n_rff, device=self.precision.device)
        self.cov.copy_(torch.linalg.inv(self.precision + self.ridge * eye))
        self.finalized.fill_(1.0)

    @torch.no_grad()
    def fit_observation_noise(self, residuals: torch.Tensor, epistemic_var: torch.Tensor):
        """
        sigma_obs^2 = max(0, Var[residual] - E[epistemic var]) on held-out data.
        See the module docstring for why this is here.
        """
        s2 = (residuals ** 2).mean() - epistemic_var.mean()
        self.sigma_obs.fill_(float(s2.clamp(min=1e-8).sqrt()))
        return float(self.sigma_obs)


class XiongForecaster(nn.Module):
    """Frozen ResNet trunk + multimodal transformer + GPA head."""

    def __init__(self, cfg, resnet: SpectralResNet):
        super().__init__()
        m = cfg["multimodal"]
        self.resnet = resnet
        self._resnet_trainable = False
        self.freeze_resnet()                      # Sec. 2.3.3: frozen backbone

        self.transformer = MultimodalTransformer(
            img_feat_dim=resnet.feat_dim,
            d_model=m["d_model"], n_heads=m["n_heads"], e_layers=m["e_layers"],
            d_layers=m["d_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
            n_temporal=len(cfg["data"]["temporal_idx"]),
            label_len=m["label_len"], pred_len=cfg["data"]["pred_len"],
            spectral_c=m.get("spectral_c", 6.0),
            use_spectral_norm=m.get("use_spectral_norm", True))

        g = cfg["gp"]
        self.gp = GPAHead(d_in=m["d_model"], n_rff=g["n_rff"],
                          kernel_scale=g.get("kernel_scale", 2.0),
                          ridge=g.get("ridge", 1e-3),
                          add_observation_noise=g.get("add_observation_noise", True))

    # -- stage control -------------------------------------------------- #
    # Sec. 2.3.3 freezes the trunk for stage 3, but stage 2 has to train it.
    # The forecaster owns the only copy, so the two stages toggle it here
    # rather than each holding their own ResNet.
    def freeze_resnet(self):
        self._resnet_trainable = False
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.resnet.eval()
        return self

    def unfreeze_resnet(self):
        self._resnet_trainable = True
        for p in self.resnet.parameters():
            p.requires_grad_(True)
        return self

    def train(self, mode=True):
        super().train(mode)
        if not self._resnet_trainable:
            self.resnet.eval()                    # keep BN frozen too
        return self

    def features(self, frames, z_hist, u_hist, u_future, img_feats=None):
        if img_feats is None:
            with torch.no_grad():
                img_feats = self.resnet.encode_sequence(frames)
        return self.transformer(img_feats, z_hist, u_hist, u_future)

    def stage_parameters(self, stage):
        """Parameters the optimiser should see for a given training stage."""
        if stage == "resnet":
            return list(self.resnet.parameters())
        if stage == "multimodal":
            return list(self.transformer.parameters()) + list(self.gp.parameters())
        raise ValueError("unknown stage for the forecaster: %s" % stage)

    def forward(self, frames, z_hist, u_hist, u_future, with_variance=False,
                img_feats=None):
        h = self.features(frames, z_hist, u_hist, u_future, img_feats=img_feats)
        return self.gp(h, with_variance=with_variance)

    def summary(self):
        resnet_state = "trainable" if self._resnet_trainable else "frozen"
        return (f"resnet {count_params(self.resnet)} ({resnet_state}) | "
                f"transformer {count_params(self.transformer)} | "
                f"gp {count_params(self.gp)}")


# =========================================================================== #
# container: one nn.Module holding every trainable piece
# =========================================================================== #
class XiongNet(nn.Module):
    """
    Wraps the whole method in a single module so the repo's generic checkpoint
    path (`checkpoint['state_dict']` -> `module.model.load_state_dict(...)`,
    main.py) round-trips all three stages at once.

    Consequence worth knowing: the stage-3 checkpoint contains the trained video
    predictor and ResNet as well, so `forecast()` needs exactly one checkpoint,
    not three.

      .video               StochasticVideoPredictor   (stage 1)
      .forecaster.resnet   SpectralResNet             (stage 2)
      .forecaster.transformer / .gp                   (stage 3)
    """

    def __init__(self, xcfg):
        super().__init__()
        from .video_prediction import StochasticVideoPredictor
        rc = xcfg["resnet"]
        resnet = SpectralResNet(in_ch=xcfg["data"]["img_channels"], base=rc["base"],
                                layers=tuple(rc["layers"]), feat_dim=rc["feat_dim"],
                                spectral_c=rc["spectral_c"],
                                use_spectral_norm=rc["use_spectral_norm"])
        self.video = StochasticVideoPredictor(xcfg)
        self.forecaster = XiongForecaster(xcfg, resnet)

    def stage_parameters(self, stage):
        if stage == "video":
            return list(self.video.parameters())
        return self.forecaster.stage_parameters(stage)

    def set_stage(self, stage):
        """requires_grad reflects the active stage only; everything else frozen."""
        for p in self.parameters():
            p.requires_grad_(False)
        if stage == "resnet":
            self.forecaster.unfreeze_resnet()
        else:
            self.forecaster.freeze_resnet()
        for p in self.stage_parameters(stage):
            p.requires_grad_(True)
        return self
