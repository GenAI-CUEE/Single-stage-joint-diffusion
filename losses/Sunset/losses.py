import torch
import torch.nn as nn
from torch.autograd import Variable

# # Check GPUs
# devCount = 0
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device() if torch.cuda.is_available() else 0
if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")

class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, y_pred, y_gt):

        return self.loss(y_pred, y_gt)

 

class MAE_Loss(nn.Module):
    def __init__(self):
        super(MAE_Loss, self).__init__()

        self.loss = nn.L1Loss()

    def forward(self, y_pred, y_gt):

        return self.loss(y_pred, y_gt)
