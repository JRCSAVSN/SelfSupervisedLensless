import torch
from torch import nn

from .util import pad_zeros_torch, crop


class Forward_Model(nn.Module):
    def __init__(self, psf: torch.Tensor, cuda_device=torch.device("cuda")):
        super(Forward_Model, self).__init__()

        self.HEIGHT = psf.shape[-2]
        self.WIDTH = psf.shape[-1]

        self.PAD_HEIGHT = int(self.HEIGHT // 2)  # Pad size
        self.PAD_WIDTH = int(self.WIDTH // 2)  # Pad size

        pad_h = torch.nn.functional.pad(psf, (self.PAD_WIDTH, self.PAD_WIDTH, self.PAD_HEIGHT, self.PAD_HEIGHT))
        otf_h = torch.fft.fft2(pad_h)

        self.H = otf_h.to(cuda_device)

    def _convolve_H(self, x):
        X = torch.fft.fft2(x)  # (n,c,h,w)
        HX = self.H * X  # (n,c,h,w)
        tmp = torch.fft.ifft2(HX)  # (n,c,h,w)
        tmp = torch.fft.ifftshift(tmp, dim=(-2, -1))  # (n,c,h,w)
        return tmp.real.float()  # (n,c,h,w)

    def forward_zero_pad(self, in_image):
        """
        Convolution operation is performed after zero padding.

        :param in_image: np.ndarray
            input image (n,c,h,w)
        :return:
        """
        output = self._convolve_H(pad_zeros_torch(self, in_image))
        cropped_output = crop(self, output)
        return cropped_output

    def forward(self, in_image):
        """
        Convolution operation is performed without zero padding.

        :param in_image: np.ndarray
            input image (n,c,h,w) h,w must be padded size.
        :return:
        """
        output = self._convolve_H(in_image)
        cropped_output = crop(self, output)
        return cropped_output