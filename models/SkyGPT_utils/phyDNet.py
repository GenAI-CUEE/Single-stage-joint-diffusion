# Adapted from PhyDNet (github.com/vincent-leguen/PhyDNet, MIT). See NOTICE.
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
        self.F.add_module('bn1',   nn.GroupNorm(7, hidden_dim))
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

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dims: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of PhyDNet layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, 
                batch_first=True, bias=True,  return_all_layers=False, pred_length=None):
        super(PhyCell, self).__init__()
 

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dims = self._extend_for_multilayer(hidden_dims, num_layers)
        if not len(kernel_size) == len(hidden_dims) == num_layers:
            raise ValueError('Inconsistent list length.')
 
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.kernel_size = kernel_size 
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.pred_length = pred_length
             
        cell_list = []
        for i in range(0, self.num_layers):
            cell_list.append(PhyDCell(input_dim=input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))                                     
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # batch, time, channel, H, W
        b, _, _, h, w = input_tensor.size()
 
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        if self.pred_length is None:
            seq_len = input_tensor.size(1)
        else:
            seq_len = self.pred_length
 
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx] 
            output_inner = []
            for t in range(seq_len): 
                h = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h)  
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  
            layer_output_list.append(layer_output) 

        if not self.return_all_layers: 
            layer_output_list = layer_output_list[-1] # layer_output_list[-1:] 

        return layer_output_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
