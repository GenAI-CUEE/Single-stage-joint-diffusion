import torch
import torch.nn as nn
from torch.autograd import Variable
from losses.PhyDNet.constrain_moments import K2M
import pdb

class Generative_loss(nn.Module):
    def __init__(self, beta_generative):
        super(Generative_loss, self).__init__()  
  
        self.criterion_mae = nn.L1Loss()   
        self.beta_generative = beta_generative   

    def forward(self, Y_prediction, Y_groundtruth):  
 
        loss = self.beta_generative * self.criterion_mae(Y_prediction, Y_groundtruth)  

        return loss  
    
    
class Moment_regularization(nn.Module):
    def __init__(self, beta_regularization):
        super(Moment_regularization, self).__init__()  

        self.beta_regularization = beta_regularization 

        self.constraints = torch.zeros((49,7,7)) 
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                self.constraints[ind,i,j] = 1
                ind +=1    

        self.criterion_mse = nn.MSELoss() 

        self.k2m = K2M([7,7]) 


    def forward(self, generator):
        loss = 0
        
        for b in range(0, generator.phycell.cell_list[0].input_dim):
            filters = generator.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
            m = self.k2m(filters.double()) 
            m  = m.float()   
            loss += self.criterion_mse(m, self.constraints.to(m.device)) # constrains is a precomputed matrix 

        loss = loss*self.beta_regularization
        return loss 




class Discriminative_loss(nn.Module):
    def __init__(self, beta_discrim):
        super(Discriminative_loss, self).__init__()  
        self.beta_discrim = beta_discrim 

    def forward(self, discr_frame_out_real,  discr_frame_out_fake):  

        # discr_frame_out_real = discriminator(Y_target).view(-1)
        # discr_frame_out_fake = discriminator(Y_prediction.detach()).view(-1)
        
        discrim_loss = self.beta_discrim * (torch.mean((discr_frame_out_real - 1)**2) + torch.mean(discr_frame_out_fake**2))
 
        return discrim_loss   


    