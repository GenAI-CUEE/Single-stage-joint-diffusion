import random

import torch
import torch.nn as nn


from models.PhyDNet_utils.ConvLSTM_utils import ConvLSTM
from models.PhyDNet_utils.PhyDNet_utils import PhyCell
from models.PhyDNet_utils.models_utils import encoder_E, encoder_specific, decoder_specific, decoder_D

class EncoderRNN(torch.nn.Module):
    """Faithful port of yuhao-nie/SkyGPT codes/video_prediction/PhyDNetGAN/
    models_v2.py::EncoderRNN. Ported the reference's per-timestep design
    (encoder_E/Ep/Er -> phycell/convcell -> decoder_Dp/Dr/D, one frame per
    call, hidden state persisting in phycell/convcell) instead of the earlier,
    buggy version, which ran the whole `seq_length`-frame sequence through the
    branches at once and bolted on a `final_layer` 1x1 conv to reproject
    `seq_length` reconstructed frames into `pred_length` channels -- a
    per-pixel channel remap, not a recurrent rollout into the future. See
    the PhyDNet.py module docstring.

    forward() drives the fixed per-step pipeline autoregressively: encode over
    the known past frames (optionally using PhyDNetGAN's own reconstruction
    instead of the true frame at some steps, per encoder_sampling_ratio --
    reference's reserve_schedule_sampling_exp), then decode pred_length future
    frames feeding the model's own previous output back in (optionally
    teacher-forced per teacher_forcing_ratio -- reference's schedule_sampling).
    Eval/forecast leave both ratios at their "always use the real/self-fed
    value" defaults, matching the reference's evaluate() exactly.
    """

    def __init__(self, configs):
        super(EncoderRNN, self).__init__()

        self.seq_length = configs["ImageSequence"]["seq_length"]
        self.pred_length = configs["ImageSequence"]["pred_length"]

        self.image_size = configs["ImageSequence"]["image_resize"]
        self.image_channel = configs["Network"]["image_channels"]

        self.hidden_dim      = configs["Network"]["hidden_dim"]
        self.half_hidden_dim = configs["Network"]["hidden_dim"] // 2


        self.phycell    = PhyCell(  input_dim=self.hidden_dim,
                                    hidden_dims= configs["Network"]["PhyCell_hidden_channels"],
                                    num_layers=len(configs["Network"]["PhyCell_hidden_channels"]),
                                    kernel_size=eval(configs["Network"]["PhyCell_kernel_size"]))

        self.convcell   = ConvLSTM( input_dim=self.hidden_dim,
                                    hidden_dims= configs["Network"]["ConvLSTM_hidden_channels"],
                                    num_layers=len(configs["Network"]["ConvLSTM_hidden_channels"]),
                                    kernel_size=eval(configs["Network"]["kernel_size"]))

        self.encoder_E  = encoder_E(nc=self.image_channel, nf=self.half_hidden_dim)   # general encoder 64x64x3 -> 32x32x32
        self.encoder_Ep = encoder_specific(nc=self.hidden_dim, nf=self.hidden_dim)    # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific(nc=self.hidden_dim, nf=self.hidden_dim)
        self.decoder_Dp = decoder_specific(nc=self.half_hidden_dim, nf=self.hidden_dim) # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific(nc=self.half_hidden_dim, nf=self.hidden_dim)
        self.decoder_D  = decoder_D(nc=self.image_channel, nf=self.half_hidden_dim)  # general decoder 32x32x32 -> 64x64x3

    def _step(self, x, first_timestep=False):
        skip_connect1, skip_connect2, encoded = self.encoder_E(x)     # 64x64x3 -> 32x32x32
        input_phys = self.encoder_Ep(encoded)                         # -> 16x16x64
        input_conv = self.encoder_Er(encoded)

        out_phys = self.phycell(input_phys, first_timestep=first_timestep)
        out_conv = self.convcell(input_conv, first_timestep=first_timestep)

        decoded_Dp = self.decoder_Dp(out_phys, skip_connect2)         # -> 32x32x(half_hidden_dim)
        decoded_Dr = self.decoder_Dr(out_conv, skip_connect2)

        concat = decoded_Dp + decoded_Dr
        return torch.sigmoid(self.decoder_D(concat, skip_connect1))   # -> 64x64x(image_channel)

    def forward(self, x_seq, y_target=None, teacher_forcing_ratio=0.0, encoder_sampling_ratio=1.0):
        seq_len = x_seq.size(1)

        encoder_input = x_seq[:, 0]
        encoder_output_image = None
        for ei in range(seq_len - 1):
            encoder_output_image = self._step(encoder_input, first_timestep=(ei == 0))
            encoder_input = x_seq[:, ei + 1] if random.random() < encoder_sampling_ratio else encoder_output_image

        # First decoder input: the last known frame, unless the encode phase's
        # last step chose to self-feed (reference's real_input_flag_encoder[-1]).
        if encoder_output_image is None or random.random() < encoder_sampling_ratio:
            decoder_input = x_seq[:, -1]
        else:
            decoder_input = encoder_output_image

        predictions = []
        for di in range(self.pred_length):
            output_image = self._step(decoder_input)
            predictions.append(output_image)
            if y_target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = y_target[:, di]
            else:
                decoder_input = output_image
        return torch.stack(predictions, dim=1)

 
