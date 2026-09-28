import torch.nn as nn


class ImageTokenizer(nn.Module):
    """Noisy latent [B,C,H,W] -> a few image tokens [B,n,dim] for the scalar side."""
    def __init__(self, in_ch, dim, n_tokens=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, 2, 1), nn.GroupNorm(8, dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),   nn.GroupNorm(8, dim), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(int(round(n_tokens ** 0.5)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z):
        h = self.pool(self.body(z))
        tok = h.flatten(2).transpose(1, 2)        # [B, n, dim]
        return self.norm(tok)
