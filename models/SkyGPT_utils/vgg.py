# Adapted from SkyGPT (github.com/yuhao-nie/SkyGPT, MIT), which builds on VideoGPT (github.com/wilson1yan/VideoGPT, MIT). See NOTICE.
import torchvision.models as models
from torch import nn

class VGGPerceptualNet(nn.Module):
    def __init__(self, layers=(3, 8, 15)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(vgg[prev:l])
            prev = l
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats
