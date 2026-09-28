"""
Pruned port of the diffusion training pipeline's conditional-DDPM base class.

The original base class (1704 lines) is the shared base for every training
stage of that pipeline (vae-only, unet-only, controlnet-only, and joint). Only
the "joint" stage's functions are ported into this repo.

Reachability was verified directly (grep of every self.<method>( call the
joint stage makes against the base class's method list): only random_latent
(which itself calls generate()), tokenize_captions, convert_to_pos_and_neg_embed,
and decode are ever called. Everything else in the base class --
train_controlnet/train_unet/train_vae + matching evaluate_*, ddpm_sampling,
controlnet_ddpm_sampling, controlnet_sampling, onepromt_sampling,
one_trainloop_controlnet, one_trainloop_unet, loading_trained_model,
_save_settings_dict -- belongs to other (unported) training stages or is
superseded by this repo's own main.py/utils/config.py infra, and is dropped.

Ported as free functions (not a class to inherit from) -- this repo's models
don't use a shared base-class pattern; models/Text2ImageJoint.py calls these
directly, matching models/Text2ImageJoint_utils/models.py's existing style.
"""

import torch


def generate(seed):
    """torch.manual_seed has a GLOBAL side effect on the default RNG -- this is
    load-bearing, not incidental: random_latent's caller (controlnet_joint_sampling)
    draws the scalar-sequence init noise `y = torch.randn(B, T, ...)` WITHOUT an
    explicit generator, so it relies on whatever this call last set as the default
    generator's state. Ported faithfully, not "cleaned up" into an explicit-generator
    call, so ensemble-member seeding in forecast() behaves identically to the source."""
    return torch.manual_seed(seed)


def random_latent(config, device, batch_size):
    return torch.randn(
        (batch_size, config["in_latent_channel"],
         config["image_resize"] // 8, config["image_resize"] // 8),
        generator=generate(config["seed"])
    ).to(device)


def _e15_row_seed(member_seed, global_sample_index):
    # Deterministic per-(member,sample) seed, independent of batch position/size --
    # collision-free for member_seed in forecast()'s 10000..10009 range and
    # global_sample_index well under 1e6 (subset is 1998 rows).
    return int(member_seed) * 1_000_000 + int(global_sample_index)


def random_latent_indexed(config, device, B, T, sample_start_index, member_seed):
    """Corrected-seeding path: ONE independent noise draw per (real dataset sample,
    member) instead of random_latent()'s single reseed-then-draw-whole-batch, which
    was found to repeat bit-identically across every batch at the same intra-batch
    position (period=4 at this repo's test_batch_size=4). Additive: random_latent()
    itself is untouched, so every existing call site (training, the frozen scoring
    artifacts) is byte-for-byte unaffected by this function's existence.

    One base seed per SAMPLE (matching sample_start_index's actual B-per-batch
    advance, exactly as random_latent_indexed_scalar below already does correctly),
    with a *1000+t spacing per frame within that sample -- collision-free by
    construction, since consecutive base seeds are 1_000_000 apart (_e15_row_seed)
    and *1000 keeps each sample's 8-frame block from ever overlapping its
    neighbours'.
    """
    C = config["in_latent_channel"]
    HW = config["image_resize"] // 8
    rows = []
    for b in range(B):
        base = _e15_row_seed(member_seed, sample_start_index + b)
        for t in range(T):
            g = torch.Generator().manual_seed(base * 1000 + t)
            rows.append(torch.randn((1, C, HW, HW), generator=g))
    return torch.cat(rows, dim=0).to(device)  # [B*T,C,H,W], b-major-then-t == reshape(B,T,...) order


def random_latent_indexed_scalar(B, T, sample_start_index, member_seed, device):
    """Corrected-seeding counterpart for the scalar trajectory's initial noise (`y` in
    models/Text2ImageJoint.py::_sample()). y's un-generator'd
    torch.randn(..., device=self.device) draws inherit random_latent()'s exact
    periodicity via torch.manual_seed's global default-RNG side effect (documented on
    generate() above) -- so y must be fixed too, or the CSI trajectory stays periodic
    even after the image latents are fixed. +1 offset keeps this stream distinct from
    random_latent_indexed's (whose seeds are ~1000x larger by construction, so the
    streams cannot collide regardless).
    """
    rows = []
    for b in range(B):
        row_seed = _e15_row_seed(member_seed, sample_start_index + b) + 1
        g = torch.Generator().manual_seed(row_seed)
        rows.append(torch.randn((1, T), generator=g))
    return torch.cat(rows, dim=0).to(device)


def tokenize_captions(tokenizer, single_promt_list: list):
    inputs = tokenizer(
        single_promt_list, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def convert_to_pos_and_neg_embed(prompt, device):
    batch, feat = prompt.shape
    positive_prompt = prompt.view(batch, 1, feat).to(device)
    negative_prompt = torch.zeros_like(positive_prompt, device=device)
    return positive_prompt, negative_prompt


def decode(vae, latents):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = torch.clip(image, min=-1, max=1)
    return image
