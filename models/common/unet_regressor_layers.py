import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Final

class Image2PV(nn.Module):
    # Image2PV receives an image [3xHxW] to a scalar output [1,]  
    def __init__(self, configs, device, image_channels=3, num_filters=32, image_size = [64,64]):
        super(Image2PV, self).__init__()
 
        self.image_channels  = configs["image_channels"]
        self.num_filters     = configs["num_filters"] 
        self.image_size      = configs["image_resize"]
        self.dropout_rate    = configs["drop_rate"]

        self.conv1x1         = nn.Conv2d(self.image_channels, self.image_channels, kernel_size=1, padding=0)

        self.conv1           = self.Conv3x3block(self.image_channels, 2*self.num_filters)
        self.conv2           = self.Conv3x3block(self.num_filters*2, self.num_filters*4)
        self.conv3           = self.Conv3x3block(self.num_filters*4, self.num_filters*4)

        self.bottleneck1     = self.BottleNeckBlock(self.num_filters*4)
        self.bottleneck2     = self.BottleNeckBlock(self.num_filters*4)

        self.upconv1         = self.Up2x2Conv3x3(self.num_filters*4, self.num_filters*4)
        self.conv4           = self.Conv3x3block(self.num_filters*8, self.num_filters*4)
        self.drop1           = nn.Dropout(p=self.dropout_rate)

        self.upconv2         = self.Up2x2Conv3x3(self.num_filters*4, self.num_filters*2) 
        self.conv5           = self.Conv3x3block(self.num_filters*4, self.num_filters*1)
        self.drop2           = nn.Dropout(p=self.dropout_rate)

        self.conv2d          = self.Conv2DRelu(self.num_filters, 1)  
        self.conv_final      = nn.Conv1d(1*self.image_size*self.image_size, 1, kernel_size=1)   

    def BottleNeckBlock(self, input_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel)
        )

    def Conv3x3block(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )
    

    def Up2x2Conv3x3(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        )
    
    def Conv2DRelu(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0), 
            nn.ReLU()
        )


    def normalize(self, image_tensor):   
        image_tensor =  image_tensor/255          # ==>  0 to +1 
        image_tensor = (image_tensor - 0.5)/0.5   # ==> -1 to +1 
        return image_tensor  
    
    def forward(self, x): 
         
        B, C, H, W  = x.shape


        out0  = self.normalize(x)  
        out1  = self.conv1x1(out0) 
        out3  = self.conv1(out1)
        out4  = F.max_pool2d(out3, kernel_size=2, stride=2)
        out6  = self.conv2(out4)
        out7  = F.max_pool2d(out6, kernel_size=2, stride=2) 
        out8  = self.conv3(out7)
        out9  = self.bottleneck1(out8)
        out10 = self.bottleneck2(out9)

        out11 = self.upconv1(out10) 
        out12 = torch.cat((out11, out6), dim=1)  
        out14 = self.conv4(out12) 
        out14 = self.drop1(out14)   

        out15 = self.upconv2(out14)  
        out16 = torch.cat((out15, out3), dim=1)   
        out18 = self.conv5(out16)
        out18 = self.drop2(out18)

        y     = self.conv2d(out18)
        y     = y.view(B, -1, 1)  
        y_out = self.conv_final(y)  
        y_out = y_out.view(B, -1) 
        return y_out


if __name__ == "__main__":
    # Assuming image_input_dim and num_filters are defined somewhere before creating the model
    model = Image2PV(image_channels=3, num_filters=12)
    x = torch.rand([4, 3, 64, 64])
    y = model(x)
