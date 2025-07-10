import torch
from torch import nn
from torchvision.transforms.functional import pad, center_crop

from .admm_utils import UnrolledADMM
from .utils import GradKeepClamp

####################################
####################################
# Encoder base class
####################################
####################################
class Encoder(nn.Module):
    '''
    Encoder class that combines physics-based deconvolution and neural-based encoding blocks. It is possible to use only deconvolution or encoding blocks by setting the corresponding parameters to None.
    Args:
        params (dict): Dictionary containing the following keys:
            - deconv_layer (nn.Module): Deconvolution layer to be used.
            - encod_layer (nn.Module): Encoder layer to be used.
    '''
    def __init__(self, params):
        super(Encoder, self).__init__()
        assert params['encod_layer'] in ['ConvEncoder', 'ResNetEncoder', 'None']

        self.encod_layer = None
        if params['encod_layer'] == 'ConvEncoder':
            self.encod_layer = ConvEncoder(params)
        elif params['encod_layer'] == 'ResNetEncoder':
            self.encod_layer = ResNetEncoder(params)

    def forward(self, x):
        
        if self.encod_layer is not None:
            x = self.encod_layer(x)
        
        return x


####################################
####################################
# Information-based encoding blocks
####################################
####################################

class ConvEncoder(nn.Module):
    def __init__(self, params):
        super(ConvEncoder, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    
class ResNetEncoder(nn.Module):
    def __init__(self, params):
        super(ResNetEncoder, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    