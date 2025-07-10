import numpy as np
import torch

def PSNR(x1, x2):
    mse = ((x1 - x2) ** 2).mean(dim=(-3, -2, -1))
    return 20 * torch.log10(1 / (mse).sqrt())