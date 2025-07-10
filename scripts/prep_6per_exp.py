import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import crop, to_tensor, to_pil_image, resize
from glob import glob
from os.path import basename, dirname
from tqdm import tqdm


from models.physicsbased import WienerDeconv, UnrolledADMM

if __name__ == '__main__':
    top = 320
    left = 530
    square_size = 1000
    psf = to_tensor(Image.open('/media/vieira/T7/YagiLabLenslessDataset/PSF/restricted/10cm.tiff'))
    psf = crop(psf, top=top, left=left, width=square_size, height=square_size)
    psf = resize(psf, 250)
    print(psf.shape)
    psf /= psf.sum()
    np.save('../psf/prototype/radial_6per.npy', psf.squeeze().numpy())


    top = 300
    left = 530
    for fname in tqdm(glob('/home/vieira/datasets/SelfSupervisedLensless/mini_FFHQ/rest_radial_6per/*.png')):
        num = int(basename(fname).split('.')[0])
        meas = to_tensor(Image.open(fname))
        meas = torch.flip(meas, dims=(-2, -1))
        meas = crop(meas, top=top, left=left, width=square_size, height=square_size)
        meas = resize(meas, 250)
        to_pil_image(meas).save(dirname(fname).replace("6per", "6per_prep") + ('/test/' if num >= 20000 else '/train/') + basename(fname))
