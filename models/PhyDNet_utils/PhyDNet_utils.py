import torch.nn as nn
import torch
import pdb

class PhyDCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(PhyDCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',   nn.GroupNorm( 7, hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        combined      = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K             = torch.sigmoid(combined_conv)
        hidden_tilde  = hidden + self.F(hidden)        # prediction
        next_hidden   = hidden_tilde + K * (x - hidden_tilde)   # correction , Haddamard product     
        return next_hidden

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.input_dim, height, width, device=self.convgate.weight.device)


class PhyCell(nn.Module):
    """Stack of PhyDCells driven ONE TIMESTEP AT A TIME -- hidden state
    (self.H) persists across forward() calls until first_timestep=True resets
    it. This matches the reference's per-step contract (yuhao-nie/SkyGPT
    codes/video_prediction/PhyDNetGAN/models_v2.py::PhyCell). The earlier
    version of this class consumed a whole given sequence in one call
    (looping `pred_length` steps reading from the *input* tensor) -- see
    the PhyDNet.py module docstring for why that never predicted the
    future.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, bias=True):
        super(PhyCell, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.H = []

        cell_list = []
        for i in range(num_layers):
            hidden_dim = hidden_dims[i] if isinstance(hidden_dims, list) else hidden_dims
            cell_list.append(PhyDCell(input_dim=input_dim,
                                       hidden_dim=hidden_dim,
                                       kernel_size=kernel_size,
                                       bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, first_timestep=False):
        batch_size, _, height, width = x.shape
        if first_timestep:
            self._init_hidden(batch_size, (height, width), x.device)
        for j, cell in enumerate(self.cell_list):
            src = x if j == 0 else self.H[j - 1]
            self.H[j] = cell(src, self.H[j])
        return self.H[-1]

    def _init_hidden(self, batch_size, image_size, device):
        self.H = [torch.zeros(batch_size, self.input_dim, *image_size, device=device)
                  for _ in range(self.num_layers)]
