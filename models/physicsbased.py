import torch
from torch import nn
from torch import Tensor
from pathlib import Path
from typing import Dict

from .admm_utils import UnrolledADMM
from .utils import GradKeepClamp #, normalize
from utils.data_utils import get_psf

####################################
####################################
# Encoder base class
####################################
####################################
class PhysicsDeconv(nn.Module):
    '''
    Encoder class that combines physics-based deconvolution and neural-based encoding blocks. It is possible to use only deconvolution or encoding blocks by setting the corresponding parameters to None.
    Args:
        params (dict): Dictionary containing the following keys:
            - deconv_layer (nn.Module): Deconvolution layer to be used.
    '''
    def __init__(self, params):
        super(PhysicsDeconv, self).__init__()
        assert params['deconv_layer'] in ['Wiener', 'DeepWiener', 'UnrolledAdmm', 'AdmmDeconv', 'PrimalDualNet', 'None']

        self.deconv_layer = None
        
        if params['deconv_layer'] == 'Wiener':
            self.deconv_layer = WienerDeconv(params)
        elif params['deconv_layer'] == 'DeepWiener':
            self.deconv_layer = DeepWienerDeconv(params)
        elif params['deconv_layer'] == 'UnrolledAdmm':
            self.deconv_layer = UnrolledAdmmDeconv(params)
        elif params['deconv_layer'] == 'AdmmDeconv':
            self.deconv_layer = AdmmDeconv(params)
        # elif params['deconv_layer'] == 'PrimalDualNet':
        #     self.deconv_layer = PrimalDualNetDeconv(params)

    def forward(self, x):
        if self.deconv_layer is not None:
            x = self.deconv_layer(x)
        return x

####################################
####################################
# Physics-based deconvolution blocks
####################################
####################################

class WienerDeconv(nn.Module):
    def __init__(self, psf, w=1, lambda_=1e-4, **kw):
        super(WienerDeconv, self).__init__()
        if isinstance(psf, (str, Path)):
            psf = get_psf(psf)
        p_f = torch.fft.fft2(torch.fft.fftshift(psf, dim=(-2, -1)))
        self.register_buffer('PSF', (w * torch.conj(p_f)) / ((torch.abs(p_f) * w) ** 2 + lambda_))
        self.w = w
        self.lambda_ = lambda_
        self.clamper = GradKeepClamp()

    def forward(self, x):
        x_f = torch.fft.fft2(torch.fft.fftshift(x, dim=(-2, -1)))
        v_f = x_f * self.PSF
        v = torch.fft.fftshift(torch.fft.ifft2(v_f), dim=(-2, -1))
        # return normalize(torch.real(v))
        return torch.real(v)
    
    def extra_repr(self):
        return f'w={self.w}, lambda_={self.lambda_}'


class DeepWienerDeconv(nn.Module):
    def __init__(self, params):
        super(DeepWienerDeconv, self).__init__()
        raise NotImplementedError()
        self.psf = params['psf']

    def forward(self, x):
        raise NotImplementedError


class UnrolledAdmmDeconv(nn.Module):
    def __init__(self, psf: str | Path | Tensor, iters: int, hyper_parameter: Dict[str, float], **kw):
        super().__init__()
        if isinstance(psf, (str, Path)):
            psf = get_psf(psf)
        self.register_buffer('psf', psf)
        self.model = UnrolledADMM(psf=psf, iters=iters, hyper_parameter=hyper_parameter)

    def forward(self, x):
        # return normalize(self.model.run(x))
        return self.model.run(x)


class AdmmDeconv(nn.Module):
    def __init__(self, psf: str | Path | Tensor, iters: int, hyper_parameter: Dict[str, float], **kw):
        super(AdmmDeconv, self).__init__()
        self.psf = psf
        self.model = UnrolledADMM(psf=psf, iters=iters, hyper_parameter=hyper_parameter)
        # Disable gradient
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model.run(x)


class UdnDeconv(nn.Module):
    pass

class PrimalDualNetDeconv(nn.Module):
    def __init__(self, params):
        super(PrimalDualNetDeconv, self).__init__()
        raise NotImplementedError()
        self.psf = params['psf']

    def forward(self, x):
        raise NotImplementedError

