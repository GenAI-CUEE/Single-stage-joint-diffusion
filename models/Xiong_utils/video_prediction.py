"""
xiong_video_prediction.py

Stage 1 of the Xiong et al. baseline: the stochastic video prediction model
(paper Sec. 2.2, Fig. 2). Three parts, exactly as in the paper:

  1. ConvGRUMotionEncoder   Sec. 2.2.1
     frame differences -> CNN -> ConvGRU -> m_{t0-1}
  2. NeuralSDEPredictor     Sec. 2.2.2, Eqs. (1)-(3)
     dX = a(X,t)dt + b(X,t)dE_t, a,b = f_theta(m); Euler-Maruyama rollout.
     *** This is the only source of ensemble diversity in the paper. ***
  3. ConditionalDiffusionGenerator  Sec. 2.2.3-2.2.4, Eqs. (4)-(8)
     eps-prediction U-Net on w^L_t = [w^0_{t-1}, S], conditioned on m_t at every
     block, trained with L1 = ||eps - eps_theta(w^L_t, w^0_{t-1}, m_t, l)||^2.

Inference is autoregressive over the horizon: the frame generated at step k
becomes the "previous frame" for step k+1, seeded by the last observed frame.

Two knobs matter more than anything else for the downstream CRPS comparison:
  sde.noise_scale   -- ensemble spread. Too small => under-dispersed baseline,
                       which would flatter the joint model unfairly. Tune it on
                       validation (see calibrate_sde_noise in train_xiong_video.py).
  ddim.eta          -- extra sampler stochasticity. Default 0.0 so that spread is
                       attributable to the SDE, as the paper intends.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.frame_utils import sinusoidal_embedding


# =========================================================================== #
# 1. ConvGRU motion feature encoder   (Sec. 2.2.1)
# =========================================================================== #
class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, ksize=3):
        super().__init__()
        pad = ksize // 2
        self.hid_ch = hid_ch
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, ksize, padding=pad)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, ksize, padding=pad)

    def forward(self, x, h):
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class ConvGRUMotionEncoder(nn.Module):
    """
    Input : clip [B, T, 3, H, W] in [-1,1]
    Output: m    [B, Cm, H/8, W/8]   the hidden state after the last difference
    """

    def __init__(self, in_ch=3, base=32, motion_ch=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, motion_ch, 4, stride=2, padding=1), nn.GroupNorm(8, motion_ch), nn.SiLU(),
        )
        self.cell = ConvGRUCell(motion_ch, motion_ch)
        self.motion_ch = motion_ch

    def forward(self, clip):
        B, T = clip.shape[:2]
        # sky-change feature matrices: differences, first frame excluded
        diffs = clip[:, 1:] - clip[:, :-1]                       # [B,T-1,3,H,W]
        Td = diffs.shape[1]
        feat = self.cnn(diffs.reshape(B * Td, *diffs.shape[2:]))  # [B*Td,Cm,h,w]
        feat = feat.reshape(B, Td, *feat.shape[1:])
        h = torch.zeros(B, self.motion_ch, feat.shape[-2], feat.shape[-1],
                        device=clip.device, dtype=feat.dtype)     # m_0 = 0
        for t in range(Td):
            h = self.cell(feat[:, t], h)
        return h                                                  # m_{t0-1}


# =========================================================================== #
# 2. Neural SDE motion predictor   (Sec. 2.2.2, Eqs. 1-3)
# =========================================================================== #
class NeuralSDEPredictor(nn.Module):
    """
    f_theta(m) -> (a, b);  m_i = m_{i-1} + a*dt + b*sqrt(dt)*xi,  xi ~ N(0, I)

    `noise_scale` multiplies b. It is the ensemble-diversity dial. b is passed
    through softplus with a negative init bias so the rollout starts almost
    deterministic and learns to open up.
    """

    def __init__(self, motion_ch=64, hidden=128, dt=1.0, noise_scale=1.0,
                 n_euler_substeps=1):
        super().__init__()
        self.dt = float(dt)
        self.noise_scale = float(noise_scale)
        self.sub = int(n_euler_substeps)
        self.net = nn.Sequential(
            nn.Conv2d(motion_ch, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, 2 * motion_ch, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():                     # drift 0, diffusion ~ softplus(-3)
            self.net[-1].bias.zero_()
            self.net[-1].bias[motion_ch:] = -3.0
        self.motion_ch = motion_ch

    def step(self, m, generator=None):
        a, b_raw = self.net(m).chunk(2, dim=1)
        b = F.softplus(b_raw) * self.noise_scale
        h = self.dt / self.sub
        for _ in range(self.sub):
            xi = torch.randn(m.shape, generator=generator, device=m.device, dtype=m.dtype)
            m = m + a * h + b * math.sqrt(h) * xi
        return m

    def rollout(self, m0, steps, generator=None):
        """m0 = m_{t0-1} -> [B, steps, Cm, h, w] for t0 .. T."""
        out, m = [], m0
        for _ in range(steps):
            m = self.step(m, generator=generator)
            out.append(m)
        return torch.stack(out, dim=1)


# =========================================================================== #
# 3. Conditional diffusion image generator   (Sec. 2.2.3-2.2.4)
# =========================================================================== #
class _ResBlock(nn.Module):
    """Conditioned on (t_emb + motion_emb) -- 'conditions to each U-Net module'."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)          # FiLM (scale, shift)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class _AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads,
                                                 H * W).unbind(1)
        att = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(C // self.heads), dim=-1)
        out = (v @ att.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)


class CDMUNet(nn.Module):
    """
    eps_theta(w^L_t, w^0_{t-1}, m_t, l).

    Input channels = 3 (noisy target frame) + 3 (previous frame, Eq. 7's
    conditioning w^0_{t-1}) [+ motion_ch if motion_spatial].
    """

    def __init__(self, img_ch=3, base=64, ch_mult=(1, 2, 2), motion_ch=64,
                 emb_dim=256, attn_res=(16,), motion_spatial=True, dropout=0.0,
                 image_size=64):
        super().__init__()
        self.motion_spatial = motion_spatial
        in_ch = 2 * img_ch + (motion_ch if motion_spatial else 0)

        self.t_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.m_mlp = nn.Sequential(nn.Linear(motion_ch, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.base = base

        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        chans = [base]
        self.downs = nn.ModuleList()
        cur, res = base, image_size
        for i, mult in enumerate(ch_mult):
            out = base * mult
            blk = nn.ModuleDict({
                "res": _ResBlock(cur, out, emb_dim, dropout),
                "attn": _AttnBlock(out) if res in attn_res else nn.Identity(),
                "down": nn.Conv2d(out, out, 3, stride=2, padding=1)
                        if i < len(ch_mult) - 1 else nn.Identity(),
            })
            self.downs.append(blk)
            chans.append(out)
            cur = out
            if i < len(ch_mult) - 1:
                res //= 2

        self.mid1 = _ResBlock(cur, cur, emb_dim, dropout)
        self.midattn = _AttnBlock(cur)
        self.mid2 = _ResBlock(cur, cur, emb_dim, dropout)

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            blk = nn.ModuleDict({
                "up": nn.Upsample(scale_factor=2, mode="nearest")
                      if i < len(ch_mult) - 1 else nn.Identity(),
                "res": _ResBlock(cur + out, out, emb_dim, dropout),
                "attn": _AttnBlock(out) if (image_size >> i) in attn_res else nn.Identity(),
            })
            self.ups.append(blk)
            cur = out

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv2d(cur, img_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_noisy, x_prev, motion, t):
        """
        x_noisy [B,3,H,W]   noisy target frame at diffusion step t
        x_prev  [B,3,H,W]   previously (generated or observed) frame
        motion  [B,Cm,h,w]  m_t from the neural SDE
        t       [B]         diffusion step index l
        """
        m_vec = motion.mean(dim=(-2, -1))
        emb = self.t_mlp(sinusoidal_embedding(t, self.base)) + self.m_mlp(m_vec)

        inp = [x_noisy, x_prev]
        if self.motion_spatial:
            inp.append(F.interpolate(motion, size=x_noisy.shape[-2:], mode="nearest"))
        h = self.stem(torch.cat(inp, dim=1))

        skips = []
        for blk in self.downs:
            h = blk["res"](h, emb)
            h = blk["attn"](h)
            skips.append(h)
            h = blk["down"](h)

        h = self.mid2(self.midattn(self.mid1(h, emb)), emb)

        for blk in self.ups:
            h = blk["up"](h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = blk["res"](torch.cat([h, s], dim=1), emb)
            h = blk["attn"](h)

        return self.out_conv(F.silu(self.out_norm(h)))


# =========================================================================== #
# diffusion schedule + DDIM sampler
# =========================================================================== #
class GaussianDiffusion:
    """Standard DDPM forward process (Eqs. 4-5) with a DDIM reverse sampler."""

    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=2e-2,
                 schedule="linear", device="cpu"):
        if schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            f = torch.cos(((t / num_train_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
            ac = f / f[0]
            betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        else:
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.T = num_train_timesteps
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas, self.alphas, self.acp = (self.betas.to(device),
                                             self.alphas.to(device),
                                             self.acp.to(device))
        return self

    def q_sample(self, x0, t, noise):
        a = self.acp[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model, x_prev, motion, shape, steps=50, eta=0.0,
                    generator=None, device="cpu", clip_denoised=True):
        """Reverse process, Eqs. (6)-(7). Returns the generated frame in [-1,1]."""
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        x = torch.randn(shape, generator=generator, device=device)
        for i, t in enumerate(ts):
            t_b = t.repeat(shape[0])
            eps = model(x, x_prev, motion, t_b)
            a_t = self.acp[t]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            if clip_denoised:
                x0 = x0.clamp(-1, 1)
            if i == len(ts) - 1:
                x = x0
                break
            a_prev = self.acp[ts[i + 1]]
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0 + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn(shape, generator=generator, device=device)
        return x


# =========================================================================== #
# full stage-1 model
# =========================================================================== #
class StochasticVideoPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v = cfg["video"]
        self.pred_len = cfg["data"]["pred_len"]
        self.encoder = ConvGRUMotionEncoder(in_ch=cfg["data"]["img_channels"],
                                            base=v["enc_base"],
                                            motion_ch=v["motion_ch"])
        self.sde = NeuralSDEPredictor(motion_ch=v["motion_ch"],
                                      hidden=v["sde_hidden"],
                                      dt=v.get("sde_dt", 1.0),
                                      noise_scale=v.get("sde_noise_scale", 1.0),
                                      n_euler_substeps=v.get("sde_substeps", 1))
        self.unet = CDMUNet(img_ch=cfg["data"]["img_channels"],
                            base=v["unet_base"],
                            ch_mult=tuple(v["unet_ch_mult"]),
                            motion_ch=v["motion_ch"],
                            emb_dim=v["emb_dim"],
                            attn_res=tuple(v.get("attn_res", [16])),
                            motion_spatial=v.get("motion_spatial", True),
                            dropout=v.get("dropout", 0.0),
                            image_size=cfg["data"]["image_size"])
        self.diffusion = GaussianDiffusion(
            num_train_timesteps=v["num_train_timesteps"],
            schedule=v.get("beta_schedule", "linear"))
        self.ddim_steps = v.get("ddim_steps", 50)
        self.ddim_eta = v.get("ddim_eta", 0.0)

    # ---------------- training loss (Eq. 8) ---------------- #
    def loss(self, x_frames, y_frames, generator=None):
        """
        x_frames [B,Tin,3,H,W], y_frames [B,Tout,3,H,W], both in [-1,1].

        One random future index k per sample (teacher forcing on w^0_{k-1}),
        one random diffusion step l per sample.
        """
        B, Tout = y_frames.shape[0], y_frames.shape[1]
        dev = x_frames.device
        self.diffusion.to(dev)

        m0 = self.encoder(x_frames)
        motions = self.sde.rollout(m0, Tout, generator=generator)   # [B,Tout,Cm,h,w]

        k = torch.randint(0, Tout, (B,), device=dev, generator=generator)
        idx = torch.arange(B, device=dev)
        target = y_frames[idx, k]                                    # [B,3,H,W]
        prev = torch.where((k == 0)[:, None, None, None],
                           x_frames[:, -1],
                           y_frames[idx, (k - 1).clamp(min=0)])
        motion = motions[idx, k]

        t = torch.randint(0, self.diffusion.T, (B,), device=dev, generator=generator)
        noise = torch.randn(target.shape, generator=generator, device=dev)
        noisy = self.diffusion.q_sample(target, t, noise)
        eps_hat = self.unet(noisy, prev, motion, t)
        return F.mse_loss(eps_hat, noise)

    # ---------------- inference ---------------- #
    @torch.no_grad()
    def sample_sequence(self, x_frames, n_samples=1, steps=None, eta=None,
                        generator=None, progress=False):
        """
        Returns [B, n_samples, Tout, 3, H, W] in [-1,1].

        Autoregressive over the horizon; each of the n_samples draws its own SDE
        path, which is what makes {W^i_{t0:T}} samples of P(w_{t0:T} | w_{1:t0}).
        """
        dev = x_frames.device
        self.diffusion.to(dev)
        steps = self.ddim_steps if steps is None else steps
        eta = self.ddim_eta if eta is None else eta
        B, _, C, H, W = x_frames.shape
        Tout = self.pred_len

        m0 = self.encoder(x_frames)
        outs = []
        it = range(n_samples)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="video samples", leave=False)
        for _ in it:
            motions = self.sde.rollout(m0, Tout, generator=generator)
            prev = x_frames[:, -1]
            frames = []
            for k in range(Tout):
                frame = self.diffusion.ddim_sample(
                    self.unet, prev, motions[:, k], (B, C, H, W),
                    steps=steps, eta=eta, generator=generator, device=dev)
                frames.append(frame)
                prev = frame
            outs.append(torch.stack(frames, dim=1))
        return torch.stack(outs, dim=1)
