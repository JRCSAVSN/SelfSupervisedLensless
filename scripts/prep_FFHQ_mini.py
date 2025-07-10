import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image, center_crop, resize, pad
from glob import glob
from PIL import Image
from tqdm import tqdm
from os import makedirs

def prep_image(fname):
    meas = to_tensor(Image.open(fname))
    meas = torch.flip(meas, (-2, -1))
    meas = center_crop(meas, (min(meas.shape[-2], meas.shape[-1]), min(meas.shape[-2], meas.shape[-1])))
    meas -= meas[0:10, 0:10].mean()
    meas.clip(0, 1)
    meas = resize(meas, (366, 366), interpolation=Image.BILINEAR)
    meas = meas.clip(0, 1)
    meas = to_pil_image(meas)
    return meas

makedirs('/home/yagilab/Desktop/datasets/Lensless2Lensless/mini_FFHQ/rest_radial_6per_prep', exist_ok=True)

if __name__ == '__main__':
    path_names = glob('/home/yagilab/Desktop/datasets/Lensless2Lensless/mini_FFHQ/rest_radial_6per/*.png')
    for fname in tqdm(path_names):
        meas = prep_image(fname)
        meas.save(fname.replace('rest_radial_6per', 'rest_radial_6per_prep'))
        
