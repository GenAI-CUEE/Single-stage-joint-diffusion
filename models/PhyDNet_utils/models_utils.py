import torch
import torch.nn as nn

        
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
        
class encoder_E(nn.Module):
    #def __init__(self, nc=1, nf=32):
    def __init__(self, nc=3, nf=32):
        super(encoder_E, self).__init__()
        # input is (3) x 64 x 64
        self.c0 = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=(1,1), stride=1, padding=0), # (32) x 64 x 64
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3,3), stride=1, padding=1), # (32) x 64 x 64
            nn.GroupNorm(16,nf),
            nn.LeakyReLU(0.2, inplace=True))
        self.c1 = dcgan_conv(nf, nf, stride=2) # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=2) # (64) x 16 x 16

    def forward(self, input):
        h0 = self.c0(input)
        h1 = self.c1(h0)  
        h2 = self.c2(h1)    
        h3 = self.c3(h2)
        return h0,h2,h3

class decoder_D(nn.Module):
    #def __init__(self, nc=1, nf=32):
    def __init__(self, nc=3, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=2) #(32) x 64 x 64
        self.upc3 = dcgan_upconv(nf*2, nf, stride=1) #(32) x 64 x 64
        self.upc4 = nn.Conv2d(in_channels=nf, out_channels=nc, kernel_size=(1,1), stride=1, padding=0)  #(nc) x 64 x 64

    def forward(self, input, skip_connect):      
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d2_c = torch.cat((d2,skip_connect),dim=1)
        d3 = self.upc3(d2_c)
        d4 = self.upc4(d3)
        
        return d4  


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=32, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=2) #(32) x 32 x 32
        self.upc3 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input, skip_connect):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d2_c = torch.cat((d2,skip_connect),dim=1)
        d3 = self.upc3(d2_c)
        return d3   
    
    
 
 