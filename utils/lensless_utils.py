import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image, pad, resize, center_crop

def shift_invariant_model(x, k):
    '''
    convolution model assuming shift invariant PSF k. k should be 2 times larger than x
    Input:
        x: input image (B, C, H, W)
        k: PSF (B, 1, H, W)
    Output:
        y: convolved image (B, C, H, W)
    '''
    assert x.shape[-2:] == k.shape[-2:], 'Natural image (x) and PSF (k) should have the same spatial dimensions. Got dimensions {} for x and {} for k'.format(x.shape[-2:], k.shape[-2:])
    k_p = pad(k, (k.shape[-1]//2, k.shape[-1]//2, k.shape[-2]//2, k.shape[-2]//2))
    x_p = pad(x, (x.shape[-1]//2, x.shape[-1]//2, x.shape[-2]//2, x.shape[-2]//2))
    k_f = torch.fft.fft2(torch.fft.fftshift(k_p, (-2, -1)))
    x_f = torch.fft.fft2(torch.fft.fftshift(x_p, (-2, -1)))
    y_f = x_f * k_f
    y = torch.fft.ifftshift(torch.fft.ifft2(y_f), (-2, -1))
    y = torch.real(y)
    return center_crop(y, (x.shape[-2], x.shape[-1]))

def shift_variant_model(x, k):
    '''
    convolution model assuming shift variant PSF k. k should be 2 times larger than x
    Input:
        x: input image (B, C, H, W)
        k: PSF (B, 1, 2H, 2W)
    Output:
        y: convolved image (B, C, H, W)
    ''' 
    assert x.shape[-2]*2 == k.shape[-2] and x.shape[-1]*2 == k.shape[-1], 'PSF should be 2 times larger than natural image. Got dimensions {} for x and {} for k'.format(x.shape[-2:], k.shape[-2:])
    y = torch.zeros_like(x)
    for i in range(x.shape[-2]):
        for j in range(x.shape[-1]):
            a, b = i+(x.shape[-2]//2), j+(x.shape[-1]//2)
            y[..., i, j] = torch.sum(x * k[..., a-(x.shape[-2]//2):a+(x.shape[-2]//2), b-(x.shape[-1]//2):b+(x.shape[-1]//2)], dim=(-1, -2))
    return y