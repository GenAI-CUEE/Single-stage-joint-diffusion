"""
temporal_joint_models.py

Option B (whole-horizon joint) -- HAND-ROLLED temporal route.

We dropped UNetMotionModel (its temporal module doesn't fit an 8x8 / (128,256,512,512)
latent -- the config-invariant "512 vs 64" failure). Instead we build on the path the
smoke test PROVED works: the existing 2D UNet + 2D ControlNet run PER FRAME over the
flattened clip [B*T, C, H, W] -- i.e. exactly one_trainloop_controlnet's call, just
batched over frames. Temporal coherence is added by a small hand-rolled TemporalRefiner
that we fully control (no library geometry to fight).

Pieces:
  * SequenceScalarDenoiser -- 8-value CSI denoiser, soft-coupled to image tokens
                              (unchanged from motion_joint_models).
  * TemporalRefiner        -- zero-init cross-frame attention producing a residual
                              correction to the per-frame eps. Identity at init.
  * handrolled_forward     -- per-frame ControlNet + per-frame 2D UNet + temporal refine.
                              Supports coupling = "aligned" (frame k sees scalar k) or
                              "soft" (every frame sees all 8 scalar tokens).

Conventions (natural, NOT the library's BCTHW quirk):
  z_clip        : [B, T, C, H, W]
  scalar_tokens : [B, T, token_dim]     from SequenceScalarDenoiser
  cond_raw      : [B, token_dim]        raw conditioning (UNet projects internally)
  controlnet_embeds : [B, Sc, cross_dim]  PROJECTED cond-only (as in _cond_embeds)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class CSIFiLMHead(nn.Module):
    """
    CSI-conditioning FiLM head (Option 1, clean-only).
    Maps a per-frame predicted clear-sky index (standardized CSI scalar) -> an embedding of
    the UNet's time-embedding dim. Attached as `unet.class_embedding`, so diffusers adds its
    output into the timestep embedding (emb = emb + class_emb) -> modulates every ResNet
    block's scale/shift (FiLM / AdaGN path), the same mechanism used for the timestep.

    Input : class_labels = per-frame CSI, shape [B*T, 1]  (detached upstream)
    Output: [B*T, emb_dim]
    Zero-init final layer so at start it contributes nothing (identity), matching the
    zero-init discipline used for TemporalRefiner.
    """
    def __init__(self, emb_dim, hidden=256, n_fourier=16):
        super().__init__()
        self.n_fourier = n_fourier
        self.register_buffer("freqs", 2.0 ** torch.arange(n_fourier).float() * math.pi)
        in_dim = 1 + 2 * n_fourier
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, emb_dim))
        nn.init.zeros_(self.net[-1].weight)   # identity at init
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, csi):                    # csi: [N,1]
        csi = csi.float().view(-1, 1)
        ff = csi * self.freqs[None, :]         # [N, n_fourier]
        feat = torch.cat([csi, torch.sin(ff), torch.cos(ff)], dim=-1)  # [N, 1+2F]
        return self.net(feat)                  # [N, emb_dim]


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


# --------------------------------------------------------------------------- #
# scalar side : 8-value sequence denoiser (unchanged)
# --------------------------------------------------------------------------- #
class _CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, ctx):
        h = self.ln_sa(x); x = x + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.ln_kv(ctx)
        x = x + self.cross_attn(self.ln_q(x), kv, kv, need_weights=False)[0]
        x = x + self.mlp(self.ln_m(x))
        return x


class SequenceScalarDenoiser(nn.Module):
    def __init__(self, token_dim, img_dim, pred_len=8, hidden=256, heads=4, depth=4):
        super().__init__()
        self.hidden = hidden
        self.value_in = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.pos = nn.Parameter(torch.randn(1, pred_len, hidden) * 0.02)
        self.cond_in = nn.Sequential(nn.Linear(token_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.t_in = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.img_proj = nn.Linear(img_dim, hidden) if img_dim != hidden else nn.Identity()
        self.blocks = nn.ModuleList([_CrossBlock(hidden, heads) for _ in range(depth)])
        self.out_eps = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.out_token = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, token_dim))

    def forward(self, y, t, cond, img_tokens):
        B, T = y.shape
        x = self.value_in(y.unsqueeze(-1)) + self.pos[:, :T]
        x = x + self.cond_in(cond).unsqueeze(1) + self.t_in(timestep_embedding(t, self.hidden)).unsqueeze(1)
        ctx = self.img_proj(img_tokens)
        for blk in self.blocks:
            x = blk(x, ctx)
        eps_y = self.out_eps(x).squeeze(-1)          # [B,T]
        tokens = self.out_token(x)                   # [B,T,token_dim]
        return eps_y, tokens


# --------------------------------------------------------------------------- #
# image side : hand-rolled temporal refiner
# --------------------------------------------------------------------------- #
class _TemporalBlock(nn.Module):
    """Self-attention over the T frame axis + MLP (per spatial location)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):                            # x: [B*H*W, T, dim]
        h = self.ln1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class TemporalRefiner(nn.Module):
    """
    Zero-init cross-frame refinement of the per-frame eps.
    Reads (noisy latent clip, per-frame eps), attends over the T frame axis at each
    spatial location, and returns eps + correction. Output projection is zero-init so
    at step 0 the module is the identity -> training starts exactly as the proven
    per-frame model and only improves.
    """
    def __init__(self, latent_ch, hidden=128, heads=4, depth=2, max_frames=8):
        super().__init__()
        self.in_proj = nn.Linear(2 * latent_ch, hidden)          # [noisy latent | eps]
        self.pos = nn.Parameter(torch.randn(1, max_frames, hidden) * 0.02)
        self.blocks = nn.ModuleList([_TemporalBlock(hidden, heads) for _ in range(depth)])
        self.out = nn.Linear(hidden, latent_ch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # identity at init

    def forward(self, z_clip, eps_spatial):          # both [B,T,C,H,W]
        B, T, C, H, W = z_clip.shape
        x = torch.cat([z_clip, eps_spatial], dim=2)                       # [B,T,2C,H,W]
        x = x.permute(0, 3, 4, 1, 2).contiguous().reshape(B * H * W, T, 2 * C)
        x = self.in_proj(x) + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x)
        corr = self.out(x)                                               # [B*H*W, T, C]
        corr = corr.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return eps_spatial + corr                                        # [B,T,C,H,W]


# --------------------------------------------------------------------------- #
# forward : per-frame ControlNet + per-frame 2D UNet + temporal refine
# --------------------------------------------------------------------------- #
def handrolled_forward(unet2d, controlnet, refiner, z_clip, timesteps,
                       scalar_tokens, cond_raw, controlnet_embeds, control_image,
                       coupling="aligned", use_controlnet=True, csi=None):
    """
    z_clip          : [B, T, C, H, W]
    timesteps       : [B]
    scalar_tokens   : [B, T, token_dim]   from SequenceScalarDenoiser
    cond_raw        : [B, token_dim]      RAW conditioning (2D UNet projects internally)
    controlnet_embeds : [B, Sc, cross_dim]  PROJECTED cond-only (as _cond_embeds gives)
    control_image   : [B, Cc, Hpix, Wpix]  spatial control = most-recent input frame i
    coupling        : "aligned" (frame k sees scalar k) | "soft" (frame sees all T)
    -> eps : [B, T, C, H, W]
    """
    B, T, C, H, W = z_clip.shape
    td = scalar_tokens.shape[-1]
    z_flat = z_clip.reshape(B * T, C, H, W)
    t_flat = timesteps.repeat_interleave(T)

    # --- build per-frame UNet context (RAW; the 2D UNet projects internally) ---
    cond_tok = cond_raw[:, None, None, :].expand(B, T, 1, td)             # [B,T,1,td]
    if coupling == "aligned":
        scal = scalar_tokens[:, :, None, :]                              # [B,T,1,td]
    else:  # soft: every frame sees all T scalar tokens
        scal = scalar_tokens[:, None, :, :].expand(B, T, T, td)          # [B,T,T,td]
    ctx = torch.cat([cond_tok, scal], dim=2)                             # [B,T,S,td]
    ctx_flat = ctx.reshape(B * T, ctx.shape[2], td)                      # [B*T,S,td] RAW

    # --- per-frame ControlNet (cond-only, projected) ---
    down_res, mid_res = None, None
    if use_controlnet:
        ctrl_ctx = controlnet_embeds.repeat_interleave(T, dim=0)         # [B*T,Sc,cross_dim]
        ctrl_img = control_image.repeat_interleave(T, dim=0)            # [B*T,Cc,Hpix,Wpix]
        down_res, mid_res = controlnet(
            z_flat, t_flat, ctrl_ctx, controlnet_cond=ctrl_img, return_dict=False)
        down_res = [r for r in down_res]

    # --- CSI-FiLM: per-frame predicted CSI -> class_labels (added into timestep emb) ---
    # csi: [B,T] (detached upstream). Flatten to [B*T,1]; None -> no conditioning.
    csi_labels = None
    if csi is not None:
        csi_labels = csi.reshape(B * T, 1).to(z_flat.dtype)

    # --- per-frame 2D UNet (this is the proven one_trainloop_controlnet call) ---
    eps_flat = unet2d(
        z_flat, t_flat, encoder_hidden_states=ctx_flat,
        class_labels=csi_labels,
        down_block_additional_residuals=down_res,
        mid_block_additional_residual=mid_res,
        return_dict=False)[0]
    eps_spatial = eps_flat.reshape(B, T, C, H, W)

    # --- temporal coherence (zero-init residual; identity at start) ---
    eps = refiner(z_clip, eps_spatial)
    return eps                                                          # [B,T,C,H,W]
