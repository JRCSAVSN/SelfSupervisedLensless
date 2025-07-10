from torch.autograd import Function
import torch

class GradKeepClamp(Function):
    @staticmethod
    def forward(ctx, i, min, max):
        result = torch.clamp(i, min, max)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def normalize(x, eps=1e-7):
    """
    Args:
        x: torch.Tensor containing non-normalized images of shape BxCxHxW
    Returns:
        x: normalized each image (CxHxW) to range 0 to 1
    """

    x = x - x.amin(dim=(1, 2, 3), keepdim=True)[0]
    x = x / (x.amax(dim=(1, 2, 3), keepdim=True)[0] + eps)

    return x