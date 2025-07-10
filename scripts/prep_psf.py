import torch
from torchvision.transforms.functional import to_tensor, to_pil_image, resize, center_crop, pad, crop
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.physicsbased import UnrolledAdmmDeconv, WienerDeconv

HEIGHT_OFFSET = 264 # 440
WIDTH_OFFSET = 264 # 440

if __name__ == '__main__':
    psf_path = '/home/yagilab/Desktop/datasets/PSF/restricted/10cm.tiff'
    centering_offset = -100
    scale_factor = 4
    # psf = to_tensor(Image.open(psf_path).convert('RGB')).float()
    psf = to_tensor(Image.open(psf_path)).float()
    psf = center_crop(psf, (min(psf.shape[-2], psf.shape[-1]), min(psf.shape[-2], psf.shape[-1])))
    print(psf.shape)
    psf = resize(psf, (366, 366), interpolation=Image.BILINEAR)
    psf -= psf[0:10, 0:10].mean()
    # np.save('psf/prototype/rest_random.npy', psf.squeeze().numpy())
    psf /= psf.sum()
    psf = psf.cuda()

    # meas = to_tensor(Image.open('/home/yagilab/Desktop/datasets/qrcode_simple_V3_prep/rest_radial/train/10_1.png'))
    meas = to_tensor(Image.open('/home/yagilab/Desktop/datasets/Lensless2Lensless/mini_FFHQ/rest_radial_6per/00000.png'))
    meas = torch.flip(meas, (-2, -1))
    meas = center_crop(meas, (min(meas.shape[-2], meas.shape[-1]), min(meas.shape[-2], meas.shape[-1])))
    meas -= meas[0:10, 0:10].mean()
    meas.clip(0, 1)
    meas = resize(meas, (366, 366), interpolation=Image.BILINEAR)
    meas -= meas.min()
    meas /= meas.max()
    meas = meas.cuda()
    print(meas.min(), meas.max())

    params = {}
    params['psf'] = psf
    params['iters'] = 100
    # params['hyper_parameter'] = {'mu1': 1e-6, 'mu2': 1e-5, 'mu3': 4e-5, 'tau': 0.0001}
    # params['hyper_parameter'] = {'mu1': 1e-4, 'mu2': 1e-5, 'mu3': 1e-5, 'tau': 1e-7}
    params['hyper_parameter'] = {'mu1': 5e-2, 'mu2': 1e-3, 'mu3': 1e-3, 'tau': 3e-7}
    params['device'] = 'cuda'
    admm_encoder = UnrolledAdmmDeconv(**params)
    admm_encoder = admm_encoder.cuda()

    params = {}
    params['psf'] = psf
    params['device'] = 'cuda'
    params['w'] = 1.0
    params['lambda_'] = 1e-4
    wiener_encoder = WienerDeconv(**params)
    wiener_encoder = wiener_encoder.cuda()

    with torch.no_grad():
        out = admm_encoder(meas.unsqueeze(0))
        # out = out.clamp(0, 1)
        out -= out.min()
        out /= out.max()

        out_w = wiener_encoder(meas.unsqueeze(0))
        out_w -= out_w.min()
        out_w /= out_w.max()


    plt.subplot(1, 4, 1)
    plt.imshow(psf.squeeze().cpu() / psf.cpu().max(), cmap='gray')

    plt.subplot(1, 4, 2)
    plt.imshow(meas.squeeze().permute(1, 2, 0).cpu(), cmap='gray')

    plt.subplot(1, 4, 3)
    plt.imshow(out.squeeze().permute(1, 2, 0).cpu(), cmap='gray')

    plt.subplot(1, 4, 4)
    plt.imshow(out_w.squeeze().permute(1, 2, 0).cpu(), cmap='gray')

    plt.show()