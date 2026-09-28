"""
Loader for the two legacy "joint" checkpoints (context_joint-0959.pth /
context_joint-0954.pth, copied into
results/Text2ImageJoint-{X0HAT,CLEAN}_pretrained/seed1/checkpoints/).

Restores vae_state_dict along with the rest of the checkpoint. An earlier
version of this loader skipped it, on the theory that the VAE is frozen during
the joint stage
(joint_parameters() never returns VAE params; set_training() always calls
vae.eval(), never .train()) so a fresh stable-diffusion-v1-5 load would be
identical. That was wrong: the checkpoint's own vae_state_dict differs from the
stock SD1.5 VAE by ~2.3% relative weight norm (verified by direct comparison) --
it was fine-tuned at an earlier training stage and simply carries through,
unchanged, into every later checkpoint. The whole UNet/ControlNet/scalar
pipeline was trained against THAT VAE's specific latent space, not stock
SD1.5's -- decoding through the wrong VAE produced plausible-looking losses
(the direct CSI/GHI head barely degraded) but garbage pixels (confirmed
visually: a flat, structureless blob instead of a recognizable sky image, at
PSNR ~12 / LPIPS ~0.84 instead of the reference ~20+ PSNR / ~0.06 LPIPS).

This is a bespoke, model-owned loader (mirrors Xiong.Model._warmstart), kept
entirely separate from main.py's own generic resume mechanism
(_get_previous_training / EarlyStopping.save_checkpoint): the legacy
checkpoint's multi-key schema (unet_state_dict, controlnet_state_dict, ...) is
not this repo's own single state_dict format, so it must never be discovered
by main.py's *.pth glob over a live run's own checkpoints/ folder.
"""

import torch


def load_pretrained_joint(model, vae, optimizer, ckpt_path, device,
                           resume_optimizer=False, strict_unet=True):
    """
    model      : the _JointNet instance (unet/controlnet/image_tokenizer/
                 scalar_denoiser/temporal_refiner/token_src)
    vae        : the frozen AutoencoderKL (sibling attribute, not part of `model`
                 -- see module docstring; restored here from the checkpoint's own
                 vae_state_dict, never trained afterward)
    optimizer  : Model.optimizer, or None at inference time
    strict_unet: False when joint.csi_condition is True (CSIFiLMHead adds a
                 unet.class_embedding submodule absent from checkpoints trained
                 without it) -- mirrors the source's
                 `strict=not config["joint"]["csi_condition"]`.

    Returns (start_epoch, step) read from the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vae.load_state_dict(ckpt["vae_state_dict"])

    missing, unexpected = model.unet.load_state_dict(
        ckpt["unet_state_dict"], strict=strict_unet)
    if missing or unexpected:
        print("[Text2ImageJoint checkpoint_io] unet load -- missing: %s | unexpected: %s"
              % (missing, unexpected))

    model.controlnet.load_state_dict(ckpt["controlnet_state_dict"])
    model.image_tokenizer.load_state_dict(ckpt["image_tokenizer_state_dict"])
    model.scalar_denoiser.load_state_dict(ckpt["scalar_denoiser_state_dict"])
    model.temporal_refiner.load_state_dict(ckpt["temporal_refiner_state_dict"])
    model.token_src.load_state_dict(ckpt["token_src_state_dict"])

    if resume_optimizer and optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError) as e:
            print("[Text2ImageJoint checkpoint_io] optimizer load skipped (%s) -- "
                  "expected if joint.freeze_image_side changed the param groups "
                  "since this checkpoint was saved." % type(e).__name__)

    start_epoch = ckpt["epoch"] + 1
    step = ckpt.get("step", 0)
    return start_epoch, step
