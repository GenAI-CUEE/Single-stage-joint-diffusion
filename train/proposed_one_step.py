"""
One denoising step of the proposed model (the deployed "x0hat" arm), on a
single random batch. Builds the real model at random initialization (no
checkpoint is loaded) and checks its training step's gradient topology: the
irradiance loss must reach the image tokenizer but must NOT reach the UNet,
because x0hat is built from a DETACHED copy of the UNet's own noise
prediction (models/Text2ImageJoint.py::_train_step, arm=="x0hat").

This is the same computation _train_step performs for that arm, kept here
rather than called through that method only so this script can backward()
the irradiance loss in isolation -- _train_step itself detaches both
component losses before returning them, since a real training step only
needs their (already backpropagated) combined sum.

Run: python train/proposed_one_step.py
"""
import sys
import pathlib

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from models.Text2ImageJoint import Model
from models.Text2ImageJoint_utils.temporal_joint_models import handrolled_forward

CONFIG = "configs/joint_x0hat_seed1.yaml"


def main():
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    # Random-init demo, not a real warm start.
    config["Text2ImageJoint"]["Network"]["pretrained_ckpt_dir"] = None
    config["Text2ImageJoint"]["Network"]["pretrained_ckpt_file"] = None

    model = Model(config)
    model.set_training()
    net, dev = model.model, model.device

    B, T = 2, model.pred_length
    px = (model.image_channels, model.image_resize, model.image_resize)
    input_clip = torch.rand(B, T, *px, device=dev)
    ghi = torch.rand(B, T, device=dev) * 400
    iclr = (torch.rand(B, T, device=dev) * 800).clamp(min=1.0)
    prompt = torch.rand(B, 4, device=dev)  # last-observed clock/Iclr covariates, as in training

    lat_shape = (B, T, model.in_latent_channel, model.image_resize // 8, model.image_resize // 8)
    noisy_lat = torch.randn(lat_shape, device=dev)
    c, h, w = noisy_lat.shape[2:]
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (B,), device=dev).long()
    y0 = model._csi_standardize(ghi, iclr)
    noisy_y = model.scheduler.add_noise(y0, torch.randn_like(y0), timesteps)
    _, controlnet_embeds, cond_vec = model._cond_embeds(prompt)

    with torch.no_grad():
        past_lat = torch.randn((B * T, model.in_latent_channel, h, w), device=dev)

    def tok_clip(latents_bt):
        tk = net.image_tokenizer(latents_bt)
        return tk.reshape(B, T * tk.shape[1], tk.shape[-1])

    tok_clean = tok_clip(past_lat) + net.token_src.weight[1]
    control_image = input_clip[:, -1]

    _, scalar_tokens = net.scalar_denoiser(noisy_y, timesteps, cond_vec, tok_clean)
    eps_z = handrolled_forward(net.unet, net.controlnet, net.temporal_refiner, noisy_lat,
                                timesteps, scalar_tokens, cond_raw=cond_vec,
                                controlnet_embeds=controlnet_embeds, control_image=control_image,
                                coupling=model.tcfg.get("coupling", "aligned"), csi=None)

    abar = model.scheduler.alphas_cumprod.to(dev)[timesteps].view(B, 1, 1, 1, 1)
    x0hat = (noisy_lat - (1.0 - abar).sqrt() * eps_z.detach()) / abar.sqrt()
    tok_x0 = tok_clip(x0hat.reshape(B * T, c, h, w)) + net.token_src.weight[0]
    eps_y, _ = net.scalar_denoiser(noisy_y, timesteps, cond_vec, tok_x0)

    loss_ghi = F.mse_loss(eps_y, torch.randn_like(eps_y))
    loss_ghi.backward()

    tok_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in net.image_tokenizer.parameters())
    unet_untouched = all(p.grad is None for p in net.unet.parameters())

    print("image_tokenizer received gradient: %s (expected True)" % tok_has_grad)
    print("UNet received no gradient:         %s (expected True)" % unet_untouched)
    assert tok_has_grad, "image_tokenizer should receive gradient through the irradiance loss"
    assert unet_untouched, "UNet must not receive gradient through the irradiance loss (eps_z.detach())"
    print("OK: the irradiance loss reaches the image tokenizer but not the UNet.")


if __name__ == "__main__":
    main()
