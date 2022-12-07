import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T

from torch import distributions
import numpy as np
from PreActResnet import PreActResFeatureNet

class ParametricDistribution(nn.Module):
    """ Distributions for sampling transformations parameters """

    def __init__(self, **kwargs):
        super(ParametricDistribution, self).__init__()

    @property
    def volume(self):
        return self.width.norm(dim=-1)

    def rsample(self, shape):
        return self.distribution.rsample(shape)

class AmortizedParamDist_Color(nn.Module):
    """Parameters are a learnt from an amortized network

    Args:
        shape_param (tuple): Shape of the parameters to be output
    """
    # TODO: do I have to 
    def __init__(self, conv, color_layer=[-1], color_dim=[2], *args, **kwargs):#! Default color_layer=[-1], color_dim=[3]
        super(AmortizedParamDist_Color, self).__init__()

        layers = color_layer
        dims = color_dim

        # Augmentation module network
        # TODO: make the PreActNetwork a seperate file
        self.conv=torch.nn.DataParallel(conv(output_layer=layers, output_dims=dims))
       
    def parameters(self):
        return self.conv.parameters()

    def forward(self, x, max_black_ratio=1.0):
        params=self.conv(x) # output should be [N,2,1,1] for lower_bound_brightness and upper_bound_brightness
        return params
    
class Color_Uniform_Dist_ConvFeature(ParametricDistribution):
    """ U([-theta, theta]) """

    len_param = 1

    def __init__(self, param, **kwargs):
        super(Color_Uniform_Dist_ConvFeature, self).__init__()
        # param for color aug is (N, 2, 1, 1)
        # original: do we need to minus 4?
        # perturbation_range=torch.sigmoid(param-4) 
        
        perturbation_range=torch.sigmoid(param) # ([N, 2, 1, 1])
        
        max_range=[0.0, 2.0] # TODO: the range could be changed
        ranges=torch.tensor(max_range).type(perturbation_range.type()).reshape([1,-1,1,1])   
             
        perturbation_range=perturbation_range*ranges
        
        r_shape=list(perturbation_range.shape)
        r_shape[1]=int(r_shape[1]/2)
        r=torch.rand(r_shape).type(perturbation_range.type()) # uniform sampling between 0-1
        #perturbation=r*perturbation_range[:,::2]-(1-r)*perturbation_range[:,1::2]
        
        # (lower_bound - upper_bound) * torch.rand + r2 is uniformly distributed on [lower_bound, upper_bound]
        perturbation= (perturbation_range[:,::2] - perturbation_range[:,1::2])*r + perturbation_range[:,1::2]
        
        self.perturbation=perturbation
        self.entropy_every=perturbation_range.mean(dim=[2, 3])
        
    @property
    def params(self):
        return {"entropy": self.entropy_every.detach().cpu()}
    
    def rsample(self):
        return self.perturbation    #!only support one sample now
    
 
class Augmentation_Module(nn.Module):
    """Implementation of Augmentation Module, takes input x and output augmented image x' """

    def __init__(self, device='cuda'):
        super(Augmentation_Module, self).__init__()
        # only transform brightness factor
        self.transform = ['brightness',]

        # Definition of PreactResNet
        color_layer=[-1]
        color_dim=[len(self.transform)*2]

        # TODO: define PreActResNet here
        conv = PreActResFeatureNet
        
        self.get_param = AmortizedParamDist_Color(conv, color_layer=color_layer, color_dim=color_dim,)
        self.distC_color=Color_Uniform_Dist_ConvFeature
         
    def parameters(self):
        return self.get_param.parameters()

    def forward(self, x):
        x_input=x

        param_color=self.get_param(x_input)[0]
        
        self.dist_color=self.distC_color(param_color,)
        perturbation=self.dist_color.rsample() # return a sample for brightness factor
        
        # make sure the brightness factor is greater than 0
        for val in perturbation.flatten().tolist():
            assert val > 0, f'{val} is less than 0'
        x = x * perturbation

        return x, perturbation
        
        
if __name__=='__main__':
    pass