"""
models/Text2ImageJoint.py

Diffusion-based joint image + GHI forecaster: the proposed model. Wrapped in
the same Model interface the rest of this repo uses (see models/Xiong.py, the
closest existing precedent -- another diffusion-based branch already
following this exact contract):

    Model(configs)          built by main.py
      .num_param            logged by main.py
      .optimizer            wrapped by train.py's ReduceLROnPlateau
      .model                nn.Module (_JointNet); main.py's generic checkpoint
                            path does module.model.load_state_dict(ckpt['state_dict'])
      .train(dataset, config, epoch)                 -> dict, per-epoch training
      .evaluation(dataset, config, epoch, mode)      -> dict, convergence monitor
      .forecast(dataset, config, epoch, mode="test") -> dict, TEST-set stats

The checkpoint format reconciliation (legacy multi-key .pth loaded via
Text2ImageJoint_utils/checkpoint_io.py, kept separate from main.py's own
resume mechanism) is documented at that module.

TWO ARMS, ONE CONFIG KEY
-------------------------
`Text2ImageJoint.Network.scalar_image_source` selects the arm (x0hat | clean |
concat | noisy), exactly mirroring the source's `joint.scalar_image_source`.
Both shipped checkpoints (context_joint-0959.pth = x0hat, context_joint-0954.pth
= clean) are the SAME class/architecture, differing only by this config key and
which pretrained checkpoint is warm-started from.

forecast() IS THE COMPARISON PATH
-----------------------------------
Like Xiong's and SkyGPT's, it targets the genuinely-future label_y and is the
only method whose numbers should appear in the paper. controlnet_joint_sampling
(here: _sample()) jointly produces BOTH outputs -- image latents and the CSI/GHI
sequence -- in one call, so forecast() ensembles BOTH per member, matching
Xiong's two-framing (a)/(b) structure:

  (a) direct CSI/GHI head -- comparable to Xiong's GHI[...] (its GPA head).
  (b) frames -> the same frozen U-Net regressor SkyGPT/Xiong use -- comparable
      to Xiong's RegGHI[...] and to SkyGPT's ens_ghi/Reg[RMSE].

This is a deliberate deviation from the source repo's own two-script split
(run_ensemble.py ensembles ONLY the direct head; evaluate_joint.py scores frame
quality from a SINGLE, non-ensembled sample) -- see the integration doc for the
full trace and its consequence: this port's own Reg[RMSE]/LPIPS numbers will not
numerically match the source's single-sample REPRODUCE.md reference table.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers import ControlNetModel

from models.common.unet_regressor import Model as Unet_regressor
from models.common.frame_export import dump_manifest_frames
from models.common.frame_utils import normalize_frames
from models.common.metrics import (crps_ensemble_fair, pooled_mae,
                                        pooled_rmse)

from models.Text2ImageJoint_utils.models import (get_CLIPTextModel,
                                                  get_CLIPTokenizer,
                                                  get_DDPMscheduler, get_unet,
                                                  get_vae)
from models.Text2ImageJoint_utils.joint_scalar_models import ImageTokenizer
from models.Text2ImageJoint_utils.temporal_joint_models import (
    CSIFiLMHead, SequenceScalarDenoiser, TemporalRefiner, handrolled_forward)
from models.Text2ImageJoint_utils.conditional_ddpm_base import (
    convert_to_pos_and_neg_embed, decode, random_latent, random_latent_indexed,
    random_latent_indexed_scalar, tokenize_captions)
from models.Text2ImageJoint_utils.checkpoint_io import load_pretrained_joint


ARMS = ("concat", "clean", "noisy", "x0hat")


class _JointNet(nn.Module):
    """
    Plain container so main.py's / EarlyStopping's generic checkpoint path
    (module.model.load_state_dict(ckpt['state_dict']), gated on
    hasattr(model.model, 'vqvae'/'transformer')) treats Text2ImageJoint exactly
    like Xiong -- a single opaque nn.Module, not a multi-stage SkyGPT-shaped
    model. See Fork A in the integration doc.
    """
    def __init__(self, unet, controlnet, image_tokenizer, scalar_denoiser,
                temporal_refiner, token_src):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_tokenizer = image_tokenizer
        self.scalar_denoiser = scalar_denoiser
        self.temporal_refiner = temporal_refiner
        self.token_src = token_src


class Model(nn.Module):
    # ---------------------------------------------------------------- #
    def __init__(self, configs):
        super(Model, self).__init__()

        self.is_train = configs["Training"]["is_train"]
        self.tcfg = configs["Text2ImageJoint"]["Network"]
        self.arm = self.tcfg.get("scalar_image_source", "clean")
        assert self.arm in ARMS, \
            "Text2ImageJoint.Network.scalar_image_source must be one of %s (got %r)" % (ARMS, self.arm)
        # exp_valid.py guards forecast() on which_model != "VQVAE"; any non-VQVAE
        # string passes. Keep the arm in it so logs say which one ran.
        self.which_model = "Text2ImageJoint-%s" % self.arm

        self.image_channels = configs["ImageSequence"].get("image_channels", 3)
        self.image_resize = configs["ImageSequence"]["image_resize"]
        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]
        self.in_latent_channel = self.tcfg.get("in_latent_channel", 4)

        dev = "cuda:%d" % configs["Training"]["gpu_id"]
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")

        # Reproducible seed, set BEFORE any module is built -- matches the
        # source's __init__ ordering (SequenceScalarDenoiser/token_src init
        # must be reproducible too, not just the sampling noise). Also doubles
        # as the per-call sampling seed (see _sample()); forecast() overrides
        # self.seed per ensemble member, mirroring run_ensemble.py's
        # `model.config["seed"] = 10_000 + s`.
        self.seed = int(configs["Training"].get("random_seed") or self.tcfg.get("seed", 1))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.is_text_format = self.tcfg.get("is_text_format", False)
        self.scheduler = get_DDPMscheduler(self.tcfg)

        if self.is_text_format:
            self.tokenizer = get_CLIPTokenizer(self.tcfg)
            self.text_encoder = get_CLIPTextModel(pretrained=False, config=self.tcfg)
            self.text_encoder.to(self.device)

        # Built from the stock stable-diffusion-v1-5 snapshot here, then
        # OVERWRITTEN below by checkpoint_io.load_pretrained_joint's
        # vae_state_dict restore (when pretrained_ckpt_dir/file is set) -- the
        # legacy checkpoint's VAE was fine-tuned at an earlier training stage
        # and the whole UNet/ControlNet/scalar pipeline was trained against
        # THAT VAE's latent space, not stock SD1.5's. Frozen thereafter (never
        # trained during the joint stage): joint_parameters() never returns
        # VAE params, set_training() always calls vae.eval(), never .train().
        self.vae = get_vae(pretrained=True, config=self.tcfg)
        self.vae.to(self.device)
        self.vae.eval()

        unet = get_unet(pretrained=False, config=self.tcfg)
        unet.to(self.device)
        controlnet = ControlNetModel.from_unet(unet)
        controlnet.to(self.device)

        # CSI-FiLM (Option 1, clean-only ablation): attach the CSI head to the
        # free class_embedding slot so diffusers adds it into the timestep
        # embedding. Both shipped configs ship csi_condition: False.
        self.csi_condition = bool(self.tcfg.get("csi_condition", False))
        if self.csi_condition:
            emb_dim = unet.time_embedding.linear_2.out_features
            unet.class_embedding = CSIFiLMHead(emb_dim).to(self.device)
            print("[Text2ImageJoint][CSI-FiLM] attached CSIFiLMHead(emb_dim=%d), mode=%s"
                  % (emb_dim, self.tcfg.get("csi_condition_mode", "real")))

        # Eval-time-only diagnostic: same weights, same arm ("x0hat" only), the ONLY
        # difference is _sample()'s loss-bearing scalar pass reading the TRUE future
        # frame instead of x0hat's own denoised estimate. Off by default; every
        # existing config is unaffected.
        self.oracle_scalar_eval = bool(self.tcfg.get("oracle_scalar_eval", False))
        if self.oracle_scalar_eval:
            assert self.arm == "x0hat", (
                "oracle_scalar_eval is only meaningful for scalar_image_source='x0hat' "
                "(the oracle swaps x0hat's own conditioning source; other arms don't read "
                "a denoised-target estimate at all) -- got arm=%r" % self.arm)
            self.which_model = "Text2ImageJoint-ORACLE"
            print(colored("[Text2ImageJoint][ORACLE] oracle_scalar_eval=True -- the loss-bearing "
                          "scalar pass will read REAL future frames, not x0hat's own denoised "
                          "estimate. Same weights as x0hat; eval-time-only diagnostic. "
                          "NEVER a forecast baseline.", "red"))

        token_dim = unet.config.encoder_hid_dim or unet.config.cross_attention_dim
        latent_ch = self.tcfg["image_tokenizer"]["latent_channel"]
        hidden = self.tcfg["scalar_denoiser"]["scalar_hidden_dim"]
        n_img_tokens = self.tcfg["image_tokenizer"]["n_img_tokens"]

        image_tokenizer = ImageTokenizer(latent_ch, hidden, n_img_tokens).to(self.device)

        scalar_heads = self.tcfg["scalar_denoiser"]["scalar_heads"]
        scalar_depth = self.tcfg["scalar_denoiser"]["scalar_depth"]
        scalar_denoiser = SequenceScalarDenoiser(
            token_dim=token_dim, img_dim=hidden, pred_len=self.pred_length,
            hidden=hidden, heads=scalar_heads, depth=scalar_depth).to(self.device)

        # zero-init; identity at start, for whole-horizon temporal coherence
        temporal_refiner = TemporalRefiner(latent_ch=latent_ch, max_frames=self.pred_length).to(self.device)

        # 0 = noisy target frame, 1 = clean past frame
        token_src = nn.Embedding(2, hidden).to(self.device)

        self.model = _JointNet(unet, controlnet, image_tokenizer, scalar_denoiser,
                               temporal_refiner, token_src)
        self._scalar_token_dim = token_dim

        # Same frozen judge SkyGPT/Xiong use, so frames->GHI numbers are
        # directly comparable across all methods.
        configs["Regressor"]["image_resize"] = configs["ImageSequence"]["image_resize"]
        self.regressor = Unet_regressor(configs["Regressor"])
        self.regressor.model.load_state_dict(
            torch.load(configs["Regressor"]["ckpt_path"], map_location=self.device,
                      weights_only=False)["state_dict"])
        self.regressor.model.eval()

        self.num_param = sum(
            p.numel() for module in
            (unet, controlnet, scalar_denoiser, image_tokenizer, temporal_refiner, token_src)
            for p in module.parameters() if p.requires_grad)

        # unlike SkyGPT's VQVAE-token dataset, nothing here depends on model
        # state, so caching per-mode (Xiong's pattern) is pure overhead saved.
        self._ds_cache = {}
        self.start_epoch = 0
        self.step = 0

        if self.is_train:
            learn = configs["Text2ImageJoint"]["Learning"]
            self.optimizer = torch.optim.AdamW(
                self.joint_parameters(),
                lr=learn["learning_rate"],
                betas=(learn.get("adam_beta1", 0.9), learn.get("adam_beta2", 0.999)),
                weight_decay=learn.get("adam_weight_decay", 0.01),
                eps=learn.get("adam_epsilon", 1e-8))
            self.grad_clip = learn.get("grad_clip", 1.0)
            self.amp = learn.get("amp", True) and self.device.type == "cuda"
            # train.py owns a ReduceLROnPlateau over self.optimizer; nothing extra here.
        else:
            self.optimizer = None
            self.grad_clip, self.amp = 1.0, False

        # Legacy checkpoint warm start -- a knob independent of main.py's own
        # load_ckpt/version resume mechanism (see Fork A). Always available,
        # evaluated once here regardless of load_ckpt, exactly like Xiong's
        # _warmstart.
        pretrained_dir = self.tcfg.get("pretrained_ckpt_dir")
        pretrained_file = self.tcfg.get("pretrained_ckpt_file")
        if pretrained_dir and pretrained_file:
            ckpt_path = os.path.join(configs["Main_logging_path"], pretrained_dir, pretrained_file)
            resume_opt = bool(self.is_train and configs["Training"].get("resume_optimizer", False))
            self.start_epoch, self.step = load_pretrained_joint(
                self.model, self.vae, self.optimizer, ckpt_path, self.device,
                resume_optimizer=resume_opt, strict_unet=not self.csi_condition)
            print(colored("[Text2ImageJoint] warm start from %s (source epoch -> next %d). "
                          "NOTE: main.py's own epoch_start (from THIS run's own, fresh "
                          "ckpt_path) still governs train.py's loop -- exactly like Xiong's "
                          "cross-stage warm start, a new version always counts epochs from "
                          "0, this is informational only." % (ckpt_path, self.start_epoch), "cyan"))
        elif self.is_train:
            print(colored("[Text2ImageJoint] no pretrained_ckpt_dir/file set -- training "
                          "from random init.", "yellow"))

        print("[%s] Num. params = %d" % (self.which_model, self.num_param))
        print("[%s] is_train = %s" % (self.which_model, self.is_train))

    # ---------------------------------------------------------------- #
    def _dataset(self, dataset, config, mode):
        """Cached dataset construction, this repo's positional signature."""
        if mode not in self._ds_cache:
            self._ds_cache[mode] = dataset(config, self.device, mode=mode,
                                           attribute_list=["label", "image", "feature"])
        return self._ds_cache[mode]

    def set_inference(self):
        self.vae.eval()
        self.model.unet.eval()
        self.model.controlnet.eval()
        self.model.scalar_denoiser.eval()
        self.model.image_tokenizer.eval()
        self.model.temporal_refiner.eval()

    def set_training(self):
        self.vae.eval()
        self.model.unet.train()
        self.model.controlnet.train()
        self.model.scalar_denoiser.train()
        self.model.image_tokenizer.train()
        self.model.token_src.train()
        self.model.temporal_refiner.train()

    def joint_parameters(self):
        # new (fresh) modules always train
        new_params = (list(self.model.scalar_denoiser.parameters())
                      + list(self.model.image_tokenizer.parameters())
                      + list(self.model.token_src.parameters())
                      + list(self.model.temporal_refiner.parameters()))
        # staged freeze: joint.freeze_image_side: true freezes the pretrained
        # image side (train only the new modules); false (both shipped
        # checkpoints) unfreezes for joint fine-tune.
        if self.tcfg.get("freeze_image_side", False):
            for p in self.model.unet.parameters():       p.requires_grad = False
            for p in self.model.controlnet.parameters(): p.requires_grad = False
            return new_params
        for p in self.model.unet.parameters():       p.requires_grad = True
        for p in self.model.controlnet.parameters(): p.requires_grad = True
        return list(self.model.unet.parameters()) + list(self.model.controlnet.parameters()) + new_params

    # ---- clear-sky-index target ---- #
    def _csi_standardize(self, ghi, iclr):
        ghi = ghi.float().to(self.device)
        iclr = iclr.float().to(self.device).clamp(min=self.tcfg.get("iclr_floor", 1.0))
        k = (ghi / iclr).clamp(0.0, self.tcfg.get("csi_clip", 1.67))
        return (k - self.tcfg.get("csi_mean", 0.7256)) / self.tcfg.get("csi_std", 0.3822)

    def _csi_to_ghi(self, csi_std, iclr):
        iclr = iclr.float().to(self.device).clamp(min=self.tcfg.get("iclr_floor", 1.0))
        k = csi_std * self.tcfg.get("csi_std", 0.3822) + self.tcfg.get("csi_mean", 0.7256)
        return k * iclr

    def _apply_csi_mode(self, csi):
        """Apply the CSI-conditioning control mode to a *detached* predicted-CSI
        tensor [B,T]: real (pass through) | shuffle (matched marginal, wrong
        pairing) | noise (information-free capacity control) | off (zeros)."""
        mode = self.tcfg.get("csi_condition_mode", "real")
        if mode == "real":
            return csi
        if mode == "shuffle":
            return csi[torch.randperm(csi.shape[0], device=csi.device)]
        if mode == "noise":
            return torch.randn_like(csi)
        if mode == "off":
            return torch.zeros_like(csi)
        raise ValueError("csi_condition_mode must be real|shuffle|noise|off, got %r" % mode)

    # ---- conditioning ---- #
    def _cond_embeds(self, prompt):
        if self.is_text_format:
            token_ = tokenize_captions(self.tokenizer, prompt)
            text_embeddings = self.text_encoder(token_.to(self.device), return_dict=False)[0]
        else:
            text_embeddings, _ = convert_to_pos_and_neg_embed(prompt, self.device)

        if self.model.unet.encoder_hid_proj is not None and \
           self.model.unet.config.encoder_hid_dim_type == "text_proj":
            controlnet_embeds = self.model.unet.encoder_hid_proj(text_embeddings)
        else:
            controlnet_embeds = text_embeddings

        cond_vec = text_embeddings.mean(dim=1)              # [B, token_dim]
        return text_embeddings, controlnet_embeds, cond_vec

    # ---- training step (per-batch body of the source's one_trainloop_joint) ---- #
    def _train_step(self, image, conditional_image, prompt, ghi, iclr):
        target_clip = image.to(self.device)
        input_clip = conditional_image.to(self.device)
        B, T = target_clip.shape[0], target_clip.shape[1]
        px = target_clip.shape[2:]

        lat = self.vae.encode(target_clip.reshape(B * T, *px)).latent_dist.sample()
        lat = lat * self.vae.config.scaling_factor
        c, h, w = lat.shape[1:]
        lat = lat.reshape(B, T, c, h, w)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (B,), device=self.device).long()
        img_noise = torch.randn_like(lat)
        noisy_lat = self.scheduler.add_noise(lat, img_noise, timesteps)

        y0 = self._csi_standardize(ghi, iclr)
        scl_noise = torch.randn_like(y0)
        noisy_y = self.scheduler.add_noise(y0, scl_noise, timesteps)

        text_embeddings, controlnet_embeds, cond_vec = self._cond_embeds(prompt)

        with torch.no_grad():
            past_lat = self.vae.encode(input_clip.reshape(B * T, *px)).latent_dist.sample()
            past_lat = past_lat * self.vae.config.scaling_factor

        def _tok_clip(latents_bt):
            tk = self.model.image_tokenizer(latents_bt)
            return tk.reshape(B, T * tk.shape[1], tk.shape[-1])

        tok_clean = _tok_clip(past_lat) + self.model.token_src.weight[1]
        tok_noisy = _tok_clip(noisy_lat.reshape(B * T, c, h, w)) + self.model.token_src.weight[0]
        control_image = input_clip[:, -1]
        coupling = self.tcfg.get("coupling", "aligned")

        if self.arm == "x0hat":
            # OPTION 2 (bidirectional): pass #1's eps_y sources the (optional,
            # detached) CSI-FiLM condition; pass #2 (loss-bearing) reads the
            # denoised-target (x0hat) tokens. Both cross-links are detached --
            # forward-value coupling only, no gradient loop.
            eps_y_prelim, scalar_tokens = self.model.scalar_denoiser(noisy_y, timesteps, cond_vec, tok_clean)

            csi_cond = None
            if self.csi_condition:
                abar1 = self.scheduler.alphas_cumprod.to(self.device)[timesteps].view(B, 1)
                csi_x0 = (noisy_y - (1.0 - abar1).sqrt() * eps_y_prelim.detach()) / abar1.sqrt()
                csi_cond = self._apply_csi_mode(csi_x0.detach())

            eps_z = handrolled_forward(
                self.model.unet, self.model.controlnet, self.model.temporal_refiner,
                noisy_lat, timesteps, scalar_tokens,
                cond_raw=cond_vec, controlnet_embeds=controlnet_embeds,
                control_image=control_image, coupling=coupling, csi=csi_cond)

            # x0 = (x_t - sqrt(1-abar)*eps) / sqrt(abar), detached
            abar = self.scheduler.alphas_cumprod.to(self.device)[timesteps].view(B, 1, 1, 1, 1)
            x0hat = (noisy_lat - (1.0 - abar).sqrt() * eps_z.detach()) / abar.sqrt()
            tok_x0 = _tok_clip(x0hat.reshape(B * T, c, h, w)) + self.model.token_src.weight[0]
            eps_y, _ = self.model.scalar_denoiser(noisy_y, timesteps, cond_vec, tok_x0)

            loss_img = F.mse_loss(eps_z, img_noise)
            loss_ghi = F.mse_loss(eps_y, scl_noise)
            loss = loss_img + self.tcfg.get("lambda_scalar", 1.0) * loss_ghi
            return loss, loss_img.detach(), loss_ghi.detach()

        if self.arm == "concat":
            img_tokens = torch.cat([tok_noisy, tok_clean], dim=1)
        elif self.arm == "clean":
            img_tokens = tok_clean
        else:  # "noisy"
            img_tokens = tok_noisy

        eps_y, scalar_tokens = self.model.scalar_denoiser(noisy_y, timesteps, cond_vec, img_tokens)

        csi_cond = None
        if self.csi_condition:
            abar = self.scheduler.alphas_cumprod.to(self.device)[timesteps].view(B, 1)
            csi_x0 = (noisy_y - (1.0 - abar).sqrt() * eps_y) / abar.sqrt()
            csi_cond = self._apply_csi_mode(csi_x0.detach())

        eps_z = handrolled_forward(
            self.model.unet, self.model.controlnet, self.model.temporal_refiner,
            noisy_lat, timesteps, scalar_tokens,
            cond_raw=cond_vec, controlnet_embeds=controlnet_embeds,
            control_image=control_image, coupling=coupling, csi=csi_cond)

        loss_img = F.mse_loss(eps_z, img_noise)
        loss_ghi = F.mse_loss(eps_y, scl_noise)
        loss = loss_img + self.tcfg.get("lambda_scalar", 1.0) * loss_ghi
        return loss, loss_img.detach(), loss_ghi.detach()

    # ---- joint sampling (ex controlnet_joint_sampling) ---- #
    @torch.no_grad()
    def _sample(self, prompt, conditional_image, iclr, num_inference_steps=None, oracle_future=None,
                seeding_mode="periodic", sample_start_index=None):
        """Whole-horizon sampling: generates the T-frame clip + T CSI values
        jointly, in one reverse-diffusion pass. Returns (latents, standardized
        CSI, GHI) -- the two output families forecast() ensembles separately.

        oracle_future: eval-only diagnostic, [B,T,C,H,W] real future frames.
        When given (arm=="x0hat" only), the loss-bearing scalar pass reads
        tokens from the TRUE future frame instead of the denoised x0hat
        estimate -- the image branch (eps_z) is completely unaffected. None
        (default) reproduces the exact headline x0hat behaviour bit-for-bit.

        seeding_mode: "periodic" (default) is BYTE-IDENTICAL to the
        pre-existing behaviour -- every existing call site is unaffected
        unless it explicitly opts in. "corrected" requires sample_start_index
        (the running position of this batch within the current subset/test
        pass) and draws one independent noise realization per (real dataset
        sample, ensemble member) instead of random_latent()'s
        reseed-per-batch draw, which was found to be bit-identical across
        every batch at the same intra-batch position."""
        self.set_inference()
        input_clip = conditional_image.to(self.device)
        B, T = input_clip.shape[0], input_clip.shape[1]

        text_embeddings, controlnet_embeds, cond_vec = self._cond_embeds(prompt)

        rl_cfg = {"seed": self.seed, "in_latent_channel": self.in_latent_channel,
                  "image_resize": self.image_resize}
        if seeding_mode == "corrected":
            assert sample_start_index is not None, "seeding_mode='corrected' requires sample_start_index"
            latents = random_latent_indexed(rl_cfg, self.device, B, T, sample_start_index,
                                             self.seed) * self.vae.config.scaling_factor
            c, h, w = latents.shape[1:]
            latents = latents.reshape(B, T, c, h, w)
            y = random_latent_indexed_scalar(B, T, sample_start_index, self.seed, self.device)
        else:
            latents = random_latent(rl_cfg, self.device, B * T) * self.vae.config.scaling_factor
            c, h, w = latents.shape[1:]
            latents = latents.reshape(B, T, c, h, w)
            y = torch.randn(B, T, device=self.device)

        with torch.no_grad():
            past_lat = self.vae.encode(input_clip.reshape(B * T, *input_clip.shape[2:])).latent_dist.sample()
            past_lat = past_lat * self.vae.config.scaling_factor

        def _tok_clip(latents_bt):
            tk = self.model.image_tokenizer(latents_bt)
            return tk.reshape(B, T * tk.shape[1], tk.shape[-1])

        tok_clean = _tok_clip(past_lat) + self.model.token_src.weight[1]
        coupling = self.tcfg.get("coupling", "aligned")
        control_image = input_clip[:, -1]

        # Oracle diagnostic: encode the TRUE future frames once (fixed across the whole reverse
        # trajectory) -- mirrors Text2Image_oracle_test.py's controlnet_joint_sampling.
        oracle_lat = None
        if oracle_future is not None:
            with torch.no_grad():
                of = oracle_future.to(self.device)                            # [B,T,C,H,W]
                oracle_lat = self.vae.encode(of.reshape(B * T, *of.shape[2:])).latent_dist.sample()
                oracle_lat = oracle_lat * self.vae.config.scaling_factor      # [B*T,c,h,w]

        steps = num_inference_steps or self.tcfg["num_inference_steps"]
        self.scheduler.set_timesteps(steps)
        for t in self.scheduler.timesteps:
            lat_in = self.scheduler.scale_model_input(latents, timestep=t)
            t_b = t.to(self.device).reshape(1).expand(B)

            if self.arm == "x0hat":
                eps_y_prelim, scalar_tokens = self.model.scalar_denoiser(y, t_b, cond_vec, tok_clean)

                csi_cond = None
                if self.csi_condition:
                    abar1 = self.scheduler.alphas_cumprod.to(self.device)[t_b].view(B, 1)
                    csi_x0 = (y - (1.0 - abar1).sqrt() * eps_y_prelim.detach()) / abar1.sqrt()
                    csi_cond = self._apply_csi_mode(csi_x0.detach())

                eps_z = handrolled_forward(
                    self.model.unet, self.model.controlnet, self.model.temporal_refiner,
                    lat_in, t_b, scalar_tokens,
                    cond_raw=cond_vec, controlnet_embeds=controlnet_embeds,
                    control_image=control_image, coupling=coupling, csi=csi_cond)
                img_step = self.scheduler.step(eps_z, t, latents)
                x0hat = img_step.pred_original_sample
                if oracle_lat is not None:
                    tok_x0 = _tok_clip(oracle_lat) + self.model.token_src.weight[0]      # ORACLE: true future frame
                else:
                    tok_x0 = _tok_clip(x0hat.reshape(B * T, c, h, w)) + self.model.token_src.weight[0]
                eps_y, _ = self.model.scalar_denoiser(y, t_b, cond_vec, tok_x0)
                latents = img_step.prev_sample
                y = self.scheduler.step(eps_y, t, y).prev_sample
                continue

            tok_noisy = _tok_clip(lat_in.reshape(B * T, c, h, w)) + self.model.token_src.weight[0]
            if self.arm == "concat":
                img_tokens = torch.cat([tok_noisy, tok_clean], dim=1)
            elif self.arm == "clean":
                img_tokens = tok_clean
            else:
                img_tokens = tok_noisy

            eps_y, scalar_tokens = self.model.scalar_denoiser(y, t_b, cond_vec, img_tokens)

            csi_cond = None
            if self.csi_condition:
                abar = self.scheduler.alphas_cumprod.to(self.device)[t_b].view(B, 1)
                csi_x0 = (y - (1.0 - abar).sqrt() * eps_y) / abar.sqrt()
                csi_cond = self._apply_csi_mode(csi_x0.detach())

            eps_z = handrolled_forward(
                self.model.unet, self.model.controlnet, self.model.temporal_refiner,
                lat_in, t_b, scalar_tokens,
                cond_raw=cond_vec, controlnet_embeds=controlnet_embeds,
                control_image=control_image, coupling=coupling, csi=csi_cond)

            latents = self.scheduler.step(eps_z, t, latents).prev_sample
            y = self.scheduler.step(eps_y, t, y).prev_sample

        ghi_pred = self._csi_to_ghi(y, iclr)
        return latents, y, ghi_pred

    # ================================================================== #
    # 1. train()
    # ================================================================== #
    def train(self, dataset, config, epoch):
        train_dataset = self._dataset(dataset, config, "train")
        trainLoader = DataLoader(train_dataset, config["Training"]["train_batch_size"],
                                 shuffle=True, drop_last=False)

        self.set_training()
        train_loss, train_img_loss, train_ghi_loss = [], [], []
        pbar = tqdm(trainLoader)
        for step, batch in enumerate(pbar):
            image = normalize_frames(batch["image_y"].to(self.device))
            conditional_image = normalize_frames(batch["image_x"].to(self.device))
            ghi_gt = batch["label_y"].squeeze(-1)
            iclr = batch["feature_y"][:, :, 3]
            token = batch["feature_x"][:, -1, :].float()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                loss, loss_img_, loss_ghi_ = self._train_step(image, conditional_image, token, ghi_gt, iclr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.joint_parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            train_loss.append(loss.item())
            train_img_loss.append(loss_img_.item())
            train_ghi_loss.append(loss_ghi_.item())
            pbar.set_description(
                "Train[%s] EP:%d [%d] Loss:%.4f (img %.4f, ghi %.4f)"
                % (self.arm, epoch, step, sum(train_loss) / len(train_loss),
                   sum(train_img_loss) / len(train_img_loss),
                   sum(train_ghi_loss) / len(train_ghi_loss)))

        return {
            "loss_per_epoch": sum(train_loss) / len(train_loss),
            # train.py logs psnr_per_epoch unconditionally; this model has no
            # pixel-space PSNR at train time (teacher-forced noise-MSE only),
            # so log 0.0 and carry the real signal in rmse/mae, Xiong's pattern.
            "psnr_per_epoch": 0.0,
            "rmse_per_epoch": math.sqrt(sum(train_img_loss) / len(train_img_loss)),
            "mae_per_epoch": sum(train_ghi_loss) / len(train_ghi_loss),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ================================================================== #
    # 2. evaluation()  -- convergence monitor, cheap on purpose
    # ================================================================== #
    def evaluation(self, dataset, config, epoch, mode="valid"):
        """
        Deliberately cheap, per Xiong.evaluation()'s own precedent: teacher-
        forced one-step denoising loss (same shape as the training loss) over
        the FULL split drives loss_per_epoch (EarlyStopping/ReduceLROnPlateau);
        real _sample() calls are bounded to Text2ImageJoint.Network.
        eval_sample_batches batches, giving a periodic, cheap image/GHI pulse
        and populating samples_list for the tracker's image dumps.

        This is a real, documented deviation from the source's evaluate_joint
        (full-sampling every validation pass, every epoch) -- report
        forecast(), not evaluation(), for real numbers.
        """
        self.set_inference()
        valid_dataset = self._dataset(dataset, config, mode)
        validLoader = DataLoader(valid_dataset, config["Training"]["%s_batch_size" % mode],
                                 shuffle=False, drop_last=False)

        n_sample_batches = int(self.tcfg.get("eval_sample_batches", 2))
        lpips_loss_fn = LPIPS(net_type="alex", reduction="none").to(self.device)

        valid_loss, valid_img_loss, valid_ghi_loss = [], [], []
        samples_list = []
        lpips_persample = []
        ghi_sqerr_persample = []

        pbar = tqdm(validLoader)
        for input_index, batch in enumerate(pbar):
            image = normalize_frames(batch["image_y"].to(self.device))
            conditional_image = normalize_frames(batch["image_x"].to(self.device))
            ghi_gt = batch["label_y"].squeeze(-1).to(self.device)
            iclr = batch["feature_y"][:, :, 3].to(self.device)
            token = batch["feature_x"][:, -1, :].float()

            with torch.no_grad():
                loss, loss_img_, loss_ghi_ = self._train_step(image, conditional_image, token, ghi_gt, iclr)
            valid_loss.append(loss.item())
            valid_img_loss.append(loss_img_.item())
            valid_ghi_loss.append(loss_ghi_.item())

            if input_index < n_sample_batches:
                with torch.no_grad():
                    latent_, _, ghi_pred = self._sample(token, conditional_image, iclr)
                    Bc, Tc = latent_.shape[0], latent_.shape[1]
                    predict = decode(self.vae, latent_.reshape(Bc * Tc, *latent_.shape[2:]))
                    predict = predict.reshape(Bc, Tc, *predict.shape[1:])

                ghi_sqerr_persample.append((ghi_pred.detach().cpu() - ghi_gt.cpu()) ** 2)

                gnd_u8 = ((image.float() / 2 + 0.5) * 255).to(torch.uint8)
                gen_u8 = ((predict.float() / 2 + 0.5) * 255).to(torch.uint8)
                lpips_val = lpips_loss_fn(image.reshape(-1, *image.shape[2:]),
                                          predict.reshape(-1, *predict.shape[2:]))
                lpips_loss_fn.reset()
                lpips_persample.append(lpips_val.view(Bc, Tc).detach().cpu())

                samples_list.append({"predictions": gen_u8[0], "ground_truth": gnd_u8[0]})

            tag = "Valid" if mode == "valid" else "Test"
            pbar.set_description("%s[%s] EP:%d [%d] Loss:%.4f"
                                 % (tag, self.arm, epoch, input_index,
                                    sum(valid_loss) / len(valid_loss)))

        result = {
            "samples_list": samples_list,
            "loss_per_epoch": sum(valid_loss) / max(len(valid_loss), 1),
            "psnr_per_epoch": 0.0,
            "rmse_per_epoch": math.sqrt(sum(valid_img_loss) / max(len(valid_img_loss), 1)),
            "mae_per_epoch": sum(valid_ghi_loss) / max(len(valid_ghi_loss), 1),
        }
        if ghi_sqerr_persample:
            result["JointGHI[RMSE]"] = float(torch.sqrt(torch.cat(ghi_sqerr_persample, dim=0).mean()))
        if lpips_persample:
            result["lpips_per_epoch"] = float(torch.cat(lpips_persample, dim=0).mean())
        return result

    # ================================================================== #
    # 3. forecast()  -- THE test-set path; these are the numbers to report
    # ================================================================== #
    def forecast(self, dataset, config, epoch, mode="test"):
        """
        N-seeded-sample ensemble (refactored from the source's run_ensemble.py
        inline loop), scoring BOTH output families _sample() jointly produces,
        per member:

          (a) direct CSI/GHI head        -- comparable to Xiong's GHI[...]
          (b) frames -> frozen regressor -- comparable to Xiong's RegGHI[...]
              and to SkyGPT's ens_ghi/Reg[RMSE]

        Use Text2ImageJoint.Network.subset_size / subset_indices to bound cost
        -- prefer subset_indices pointing at skygpt_subset_indices.npy so this
        scores the EXACT samples SkyGPT/Xiong were scored on. Output is named
        text2imagejoint_forecast_stats_full.npz when run on the full,
        unsubsetted test set (no subset_indices/subset_size configured).
        """
        net = self.tcfg
        n_ens = int(net.get("n_ensemble", 10))
        seed_base = int(net.get("n_ensemble_seed_base", 10000))
        max_dumps = int(net.get("max_image_dumps", 8))
        iclr_floor = net.get("iclr_floor", 1.0)
        # Off-by-default -- "periodic" is BYTE-IDENTICAL to pre-existing behaviour, so
        # every existing config/frozen artifact is unaffected unless it explicitly sets
        # Text2ImageJoint.Network.seeding_mode.
        seeding_mode = net.get("seeding_mode", "periodic")

        self.set_inference()
        lpips_loss_fn = LPIPS(net_type="alex", reduction="none").to(self.device)

        test_dataset = self._dataset(dataset, config, mode)
        test_dataset, chosen = self._maybe_subset(test_dataset, config, net)
        loader = DataLoader(test_dataset, config["Training"]["%s_batch_size" % mode],
                            shuffle=False, drop_last=False)

        S_list, ENS_GHI_list, GT_list, KORG_list, ICLR_list = [], [], [], [], []
        reg_sqerr_list, reg_abserr_list = [], []
        pixel_sqerr_list, pixel_abserr_list, lpips_list = [], [], []
        index_ps = []
        pixel_sqerr_sum, pixel_abserr_sum, pixel_elem = 0.0, 0.0, 0
        sample_counter = 0
        samples_list = []

        pbar = tqdm(loader)
        for input_index, batch in enumerate(pbar):
            conditional_image = normalize_frames(batch["image_x"].to(self.device))
            target_image = normalize_frames(batch["image_y"].to(self.device))
            ghi_gt = batch["label_y"].squeeze(-1).to(self.device)
            iclr_y = batch["feature_y"][:, :, 3].to(self.device)
            token = batch["feature_x"][:, -1, :].float()

            ghi_i = batch["label_x"][:, -1, 0].float()
            iclr_i = batch["feature_x"][:, -1, 3].float().clamp(min=iclr_floor)
            k_origin = ghi_i / iclr_i

            B, T = target_image.shape[0], target_image.shape[1]
            gnd_u8 = ((target_image.float() / 2 + 0.5) * 255).to(torch.uint8)

            member_ghi, member_reg_ghi = [], []
            member_reg_sqerr, member_reg_abserr = [], []
            member_pixel_sqerr, member_pixel_abserr, member_lpips = [], [], []

            for s in range(n_ens):
                # per-member seed override -- without it every member draws an
                # identical sample (std == 0), same requirement run_ensemble.py
                # documents for controlnet_joint_sampling.
                self.seed = seed_base + s
                with torch.no_grad():
                    oracle_future = target_image if self.oracle_scalar_eval else None
                    latent_, _, ghi_pred = self._sample(token, conditional_image, iclr_y,
                                                        oracle_future=oracle_future,
                                                        seeding_mode=seeding_mode,
                                                        sample_start_index=sample_counter)
                    Bc, Tc = latent_.shape[0], latent_.shape[1]
                    predict = decode(self.vae, latent_.reshape(Bc * Tc, *latent_.shape[2:]))
                    predict = predict.reshape(Bc, Tc, *predict.shape[1:])

                    reg_ghi = self.regressor(
                        ((predict.reshape(Bc * Tc, *predict.shape[2:]).float() / 2 + 0.5) * 255)
                    ).reshape(Bc, Tc)

                member_ghi.append(ghi_pred.detach().cpu())
                member_reg_ghi.append(reg_ghi.detach().cpu())
                member_reg_sqerr.append(((reg_ghi.detach() - ghi_gt) ** 2).cpu())
                member_reg_abserr.append((reg_ghi.detach() - ghi_gt).abs().cpu())

                gen_u8 = ((predict.float() / 2 + 0.5) * 255).to(torch.uint8)
                # int16 diff: uint8 arithmetic wraps (0 - 1 -> 255)
                pdiff = gnd_u8.to(torch.int16) - gen_u8.to(torch.int16)
                pixel_sqerr_sum += float((pdiff.float() ** 2).sum())
                pixel_abserr_sum += float(pdiff.abs().sum())
                pixel_elem += pdiff.numel()
                _pd = pdiff.float().reshape(B, T, -1)
                member_pixel_sqerr.append((_pd ** 2).mean(dim=2).cpu())
                member_pixel_abserr.append(_pd.abs().mean(dim=2).cpu())

                lpips_val = lpips_loss_fn(target_image.reshape(-1, *target_image.shape[2:]),
                                          predict.reshape(-1, *predict.shape[2:]))
                lpips_loss_fn.reset()
                member_lpips.append(lpips_val.view(B, T).detach().cpu())

                if s == 0 and len(samples_list) < max_dumps:
                    samples_list.append({"predictions": gen_u8[0], "ground_truth": gnd_u8[0]})
                if s == 0:
                    gen_u8_s0 = gen_u8  # the representative single-draw frame

            # Manifest-keyed frame export (aligned, unlike evaluation()'s shuffled
            # dumps). No-op for batches that don't overlap frame_manifest.csv, and a
            # no-op entirely unless that file is present. Uses the s==0 member's
            # frame, the same representative single draw samples_list above uses.
            # Written into THIS run's own results folder (test-images/{case_id}/, one
            # subfolder per manifest case -- e.g. test-images/ramp_up_03/), not a
            # shared path -- see models/common/frame_export.py's module docstring for
            # why (silent cross-config overwrite when the same method's forecast()
            # runs more than once under different configs). dump_manifest_frames
            # itself appends "/{case_id}" to out_dir, so out_dir here is just test-images/.
            if mode == "test":
                test_images_dir = os.path.join(config["logging_version_path"], "test-images")
                dump_manifest_frames("text2imagejoint_%s" % self.arm, sample_counter, gnd_u8, gen_u8_s0,
                                      out_dir=test_images_dir)

            S_list.append(torch.stack(member_ghi, dim=1))                    # [B,M,T]
            ENS_GHI_list.append(torch.stack(member_reg_ghi, dim=1))          # [B,M,T]
            reg_sqerr_list.append(torch.stack(member_reg_sqerr, dim=1))
            reg_abserr_list.append(torch.stack(member_reg_abserr, dim=1))
            pixel_sqerr_list.append(torch.stack(member_pixel_sqerr, dim=1))
            pixel_abserr_list.append(torch.stack(member_pixel_abserr, dim=1))
            lpips_list.append(torch.stack(member_lpips, dim=1))

            GT_list.append(ghi_gt.cpu())
            KORG_list.append(k_origin.cpu())
            ICLR_list.append(iclr_y.cpu())
            index_ps.append(torch.arange(sample_counter, sample_counter + B))
            sample_counter += B

            pbar.set_description("Test[forecast] EP:%d [%d] N_ens:%d" % (epoch, input_index, n_ens))

        S = torch.cat(S_list, dim=0).numpy().astype(np.float64)              # [N,M,T]
        ENS_GHI = torch.cat(ENS_GHI_list, dim=0).numpy().astype(np.float64)  # [N,M,T]
        GT = torch.cat(GT_list, dim=0).numpy().astype(np.float64)            # [N,T]
        ICLR = np.clip(torch.cat(ICLR_list, dim=0).numpy().astype(np.float64), iclr_floor, None)
        KORG = torch.cat(KORG_list, dim=0).numpy().astype(np.float64)
        SPM = KORG[:, None] * ICLR
        reg_sqerr = torch.cat(reg_sqerr_list, dim=0).numpy()
        reg_abserr = torch.cat(reg_abserr_list, dim=0).numpy()
        pixel_sqerr = torch.cat(pixel_sqerr_list, dim=0).numpy()
        pixel_abserr = torch.cat(pixel_abserr_list, dim=0).numpy()
        lpips_arr = torch.cat(lpips_list, dim=0).numpy()
        index_all = torch.cat(index_ps, dim=0).numpy()

        result = self._forecast_metrics(S, ENS_GHI, GT, SPM, reg_sqerr, reg_abserr,
                                        pixel_sqerr, pixel_abserr, lpips_arr,
                                        pixel_sqerr_sum, pixel_abserr_sum, pixel_elem)
        result["samples_list"] = samples_list

        if mode == "test":
            suffix = "_full" if chosen is None else ""
            npz_file = os.path.join(config["ckpt_path"], "text2imagejoint_forecast_stats%s.npz" % suffix)
            np.savez_compressed(
                npz_file,
                # --- direct CSI/GHI head (framing a) -------------------------- #
                S=S.astype(np.float32), GT=GT.astype(np.float32),
                k_origin=KORG.astype(np.float32), iclr_y=ICLR.astype(np.float32),
                SPM=SPM.astype(np.float32),
                # --- frames -> shared regressor (framing b, Xiong/SkyGPT-comparable) #
                ens_ghi=ENS_GHI.astype(np.float32),
                reg_sqerr=reg_sqerr.astype(np.float32), reg_abserr=reg_abserr.astype(np.float32),
                # --- image quality, same keys SkyGPT/Xiong write --------------- #
                pixel_sqerr=pixel_sqerr.astype(np.float32), pixel_abserr=pixel_abserr.astype(np.float32),
                lpips=lpips_arr.astype(np.float32),
                pixel_mse=np.float64(pixel_sqerr_sum / pixel_elem),
                # --- joins / provenance ---------------------------------------- #
                index=index_all,
                subset_indices=(np.asarray(chosen) if chosen is not None else np.arange(GT.shape[0])),
                checkpoint=str(config["ckpt_path"]), arm=self.arm, n_ensemble=n_ens,
                oracle_scalar_eval=self.oracle_scalar_eval, seeding_mode=seeding_mode)
            print("[Text2ImageJoint forecast] saved -> %s   (S=%s, GT=%s)" % (npz_file, S.shape, GT.shape))
            print("[Text2ImageJoint forecast] cross-check: python analyze_ensemble.py %s" % npz_file)

        return result

    # ---------------------------------------------------------------- #
    def _maybe_subset(self, valid_dataset, config, net):
        """
        Prefer an explicit index file -- the only way to guarantee the SAME
        samples as SkyGPT/Xiong. Falls back to the identical stratified scheme
        (k_origin tertiles, rng seed 1234) if only a size is given, matching
        Xiong._maybe_subset exactly.
        """
        idx_file = net.get("subset_indices", None)
        if idx_file:
            path = idx_file if os.path.isabs(idx_file) else \
                os.path.join(config["Main_logging_path"], idx_file)
            chosen = np.load(path).tolist()
            print(colored("[Text2ImageJoint] scoring the %d samples from %s (matched to SkyGPT)"
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
        np.save(os.path.join(config["ckpt_path"], "text2imagejoint_subset_indices.npy"),
                np.asarray(chosen))
        print("[Text2ImageJoint] stratified subset: %d samples, k tertiles %.3f/%.3f"
              % (len(chosen), q[0], q[1]))
        return torch.utils.data.Subset(valid_dataset, chosen), chosen

    # ---------------------------------------------------------------- #
    def _forecast_metrics(self, S, ENS_GHI, GT, SPM, reg_sqerr, reg_abserr,
                          pixel_sqerr, pixel_abserr, lpips_arr,
                          pixel_sqerr_sum, pixel_abserr_sum, pixel_elem):
        """Both framings, one flat dict for the repo's CSV logger. Reuses
        models/Xiong_utils/metrics.py's pooled_rmse/pooled_mae/
        crps_ensemble_fair -- same estimator as Xiong, cross-checkable."""
        T = GT.shape[1]
        result = {}

        # ---------- image quality (same keys SkyGPT/Xiong report) ---------- #
        pixel_mse = pixel_sqerr_sum / pixel_elem
        result["psnr_per_epoch"] = 20.0 * math.log10(255.0 / math.sqrt(max(pixel_mse, 1e-12)))
        result["rmse_per_epoch"] = math.sqrt(pixel_mse)
        result["mae_per_epoch"] = pixel_abserr_sum / pixel_elem
        result["lpips_per_epoch"] = float(lpips_arr.mean())

        # per-member pixel/LPIPS errors, pooled over BOTH samples and members
        rmse_h = np.sqrt(pixel_sqerr.mean(axis=(0, 1)))
        lpips_h = lpips_arr.mean(axis=(0, 1))
        for i in range(T):
            result["PSNR_%d" % (i + 1)] = float(20.0 * np.log10(255.0 / max(rmse_h[i], 1e-8)))
            result["LPIPS_%d" % (i + 1)] = float(lpips_h[i])

        reg_rmse_h = np.sqrt(reg_sqerr.mean(axis=(0, 1)))
        reg_mae_h = reg_abserr.mean(axis=(0, 1))
        for i in range(T):
            result["Reg[RMSE]_%d" % (i + 1)] = float(reg_rmse_h[i])
            result["Reg[MAE]_%d" % (i + 1)] = float(reg_mae_h[i])
        result["Reg[RMSE]_avg"] = float(reg_rmse_h.mean())
        result["Reg[MAE]_avg"] = float(reg_mae_h.mean())

        spm_rmse = pooled_rmse(SPM, GT)
        spm_mae = pooled_mae(SPM, GT)
        if float(spm_rmse.mean()) < 1.0 or float(spm_mae.mean()) < 1.0:
            print(colored("[Text2ImageJoint] WARNING: smart-persistence RMSE is ~0 (%.3f). "
                          "FS/CRPSS are meaningless here -- on real SIRTA data expect "
                          "~123 W/m^2; check k_origin/Iclr/the split."
                          % float(spm_rmse.mean()), "red"))
        spm_rmse = np.clip(spm_rmse, 1e-3, None)
        spm_mae = np.clip(spm_mae, 1e-3, None)

        # ---------- (a) direct CSI/GHI head ---------- #
        S_mean = S.mean(axis=1)                                            # [N,T]
        single = float(np.mean([pooled_rmse(S[:, j, :], GT).mean() for j in range(S.shape[1])]))
        rmse_ens = pooled_rmse(S_mean, GT)
        crps = crps_ensemble_fair(S, GT).mean(axis=0)

        result["GHI[single-sample RMSE]"] = single
        result["GHI[ensemble RMSE]"] = float(rmse_ens.mean())
        result["GHI[SPM RMSE]"] = float(spm_rmse.mean())
        result["GHI[FS]"] = float((1.0 - rmse_ens / spm_rmse).mean())
        result["GHI[CRPS]"] = float(crps.mean())
        result["GHI[CRPSS]"] = float((1.0 - crps / spm_mae).mean())
        result["GHI[spread]"] = float(S.std(axis=1).mean())
        for i in range(T):
            result["GHI[RMSE]_%d" % (i + 1)] = float(rmse_ens[i])
            result["GHI[CRPS]_%d" % (i + 1)] = float(crps[i])

        # ---------- (b) frames -> shared regressor (Xiong/SkyGPT-comparable) -- #
        reg_mean = ENS_GHI.mean(axis=1)                                    # [N,T]
        reg_rmse_ens = pooled_rmse(reg_mean, GT)
        reg_crps = crps_ensemble_fair(ENS_GHI, GT).mean(axis=0)
        result["RegGHI[ensemble RMSE]"] = float(reg_rmse_ens.mean())
        result["RegGHI[FS]"] = float((1.0 - reg_rmse_ens / spm_rmse).mean())
        result["RegGHI[CRPS]"] = float(reg_crps.mean())
        result["RegGHI[CRPSS]"] = float((1.0 - reg_crps / spm_mae).mean())
        result["RegGHI[spread]"] = float(ENS_GHI.std(axis=1).mean())

        print("\n" + "=" * 72)
        print("TEXT2IMAGEJOINT FORECAST  arm=%s  (N=%d, %d samples)" % (self.arm, S.shape[1], GT.shape[0]))
        print("  (a) direct CSI head  : RMSE %7.1f  FS %+.3f  CRPS %6.1f  CRPSS %+.3f"
              % (result["GHI[ensemble RMSE]"], result["GHI[FS]"],
                 result["GHI[CRPS]"], result["GHI[CRPSS]"]))
        print("  (b) frames->regressor: RMSE %7.1f  FS %+.3f  CRPS %6.1f  CRPSS %+.3f"
              % (result["RegGHI[ensemble RMSE]"], result["RegGHI[FS]"],
                 result["RegGHI[CRPS]"], result["RegGHI[CRPSS]"]))
        print("      smart persistence: RMSE %7.1f" % result["GHI[SPM RMSE]"])
        print("  image: Reg[RMSE] %.1f  LPIPS %.4f  Pixel-RMSE %.2f"
              % (result["Reg[RMSE]_avg"], result["lpips_per_epoch"], result["rmse_per_epoch"]))
        print("  NOTE: unlike the source repo's REPRODUCE.md numbers (single-sample "
              "image-quality scoring), (b)/Reg[RMSE]/LPIPS here are ensembled over "
              "all %d members -- not directly comparable to those reference numbers, "
              "see Text2ImageJoint_integration.md." % S.shape[1])
        print("=" * 72 + "\n")
        return result
