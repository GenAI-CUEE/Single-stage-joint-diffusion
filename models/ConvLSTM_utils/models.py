import torch
import torch.nn as nn
import random


class ConvLSTMCell(nn.Module):
    """Single conv-LSTM cell, one timestep. Matches yuhao-nie/SkyGPT
    codes/video_prediction/ConvLSTM/models.py::ConvLSTM_Cell."""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                               out_channels=4 * hidden_dim,
                               kernel_size=kernel_size,
                               padding=self.padding, bias=bias)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """Stack of ConvLSTMCells driven ONE TIMESTEP AT A TIME -- hidden state
    persists across forward() calls until first_timestep=True resets it. This
    is the reference's per-step contract (codes/video_prediction/ConvLSTM/
    models.py::ConvLSTM), not a "consume a whole given sequence" module. The
    earlier version of this file read all `pred_length` frames from the given
    *input* tensor every step and never fed its own predictions back in -- see
    the ConvLSTM.py module docstring.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.H, self.C = [], []
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, first_timestep=False):
        batch_size, _, height, width = x.shape
        if first_timestep:
            self._init_hidden(batch_size, (height, width), x.device)
        for j, cell in enumerate(self.cell_list):
            src = x if j == 0 else self.H[j - 1]
            self.H[j], self.C[j] = cell(src, (self.H[j], self.C[j]))
        return self.H[-1]

    def _init_hidden(self, batch_size, image_size, device):
        self.H, self.C = [], []
        for i in range(self.num_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], *image_size, device=device))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], *image_size, device=device))


class decoder_D(nn.Module):
    """1x1 conv projecting the last ConvLSTM layer's hidden state to image
    channels, matching the reference's decoder_D."""

    def __init__(self, image_channels, hidden_dim):
        super(decoder_D, self).__init__()
        self.d1 = nn.Conv2d(hidden_dim, image_channels, kernel_size=1)

    def forward(self, x):
        return self.d1(x)


class EncoderRNN(nn.Module):
    """Faithful port of the reference EncoderRNN, extended with an explicit
    autoregressive rollout: encode over the given past frames to build hidden
    state, then decode `pred_length` future frames, feeding the model's own
    previous output back in as the next step's input. Optional scheduled
    teacher forcing during training (y_target/teacher_forcing_ratio); eval and
    forecast leave y_target=None, which always self-feeds -- matching the
    reference's evaluate() exactly."""

    def __init__(self, convcell, image_channels):
        super(EncoderRNN, self).__init__()
        self.convcell = convcell
        self.decoder_D = decoder_D(image_channels, convcell.hidden_dims[-1])

    def _step(self, x, first_timestep=False):
        h_last = self.convcell(x, first_timestep=first_timestep)
        return torch.sigmoid(self.decoder_D(h_last))

    def forward(self, x_seq, pred_length, y_target=None, teacher_forcing_ratio=0.0):
        seq_len = x_seq.size(1)
        for ei in range(seq_len - 1):
            self._step(x_seq[:, ei], first_timestep=(ei == 0))
        decoder_input = x_seq[:, -1]

        predictions = []
        for di in range(pred_length):
            output_image = self._step(decoder_input)
            predictions.append(output_image)
            if y_target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = y_target[:, di]
            else:
                decoder_input = output_image
        return torch.stack(predictions, dim=1)
