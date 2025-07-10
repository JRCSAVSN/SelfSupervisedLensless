import torch.nn.functional as F
import numpy as np
import torch
import torch.optim


def flip_channels(image):
    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1] = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]
    return image_color


def pad_zeros_torch(model, x) -> torch.Tensor:
    PADDING = (model.PAD_WIDTH, model.PAD_WIDTH, model.PAD_HEIGHT, model.PAD_HEIGHT)
    return F.pad(x, PADDING, 'constant', 0)


def crop(model, x) -> torch.Tensor:
    C01 = model.PAD_HEIGHT
    C02 = model.PAD_HEIGHT + model.HEIGHT  # Crop indices
    C11 = model.PAD_WIDTH
    C12 = model.PAD_WIDTH + model.WIDTH  # Crop indices
    return x[..., C01:C02, C11:C12]


def shift(frame, y: int, x: int):
    assert x < 0 and y < 0
    h, w = frame.shape
    result = np.zeros_like(frame)
    result[y:, x:] = frame[:h - y, w - x]
    return result


def tv_loss(x):
    """
    Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
    """
    x = x.cuda()
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(dh[:, :, :-1] + dw[:, :, :, :-1])


def uint8(x):
    x = (x - x.min()) / (x.max() - x.min())
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)