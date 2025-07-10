import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image, to_tensor, crop
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from glob import glob
from os.path import basename

if __name__ == '__main__':
    
    # Prep PSF
    psf = np.array(Image.open('/home/vieira/Downloads/rest_radial.bmp')).astype(np.float32) / 255.0
    square_size = 1200
    left = 420
    top = 60
    psf = psf[top:top+square_size, left:left+square_size]
    psf = resize(torch.from_numpy(psf).unsqueeze(0), 300).squeeze()
    psf /= psf.sum()
    np.save('../psf/prototype/radial_25per.npy', psf.squeeze().numpy())

    # Prep dataset
    

    class DataPrep(Dataset):
        def __init__(self, data_path):
            self.fnames = glob(data_path + '/*.png')
        
        def __getitem__(self, idx):
            meas = to_tensor(Image.open(self.fnames[idx]))
            meas = torch.flip(meas, dims=(-1, -2))
            square_size = 1400
            left = 400
            top = 00
            meas = crop(meas, top=top, left=left, width=square_size, height=square_size)
            meas = resize(meas, 300).squeeze()
            return meas, self.fnames[idx]
        
        def __len__(self):
            return len(self.fnames)

    dset = DataPrep(data_path = '/home/vieira/datasets/SelfSupervisedLensless/mini_FFHQ/rest_radial_25per/')
    dloader = DataLoader(dset, batch_size=128, num_workers=4)

    for data, fname in tqdm(dloader):
        for i in range(data.shape[0]):
            # np.save(f'/home/vieira/datasets/SelfSupervisedLensless/mini_FFHQ/rest_radial_25per_prep/{basename(fname[i])}', to_pil_image(data[i]))
            to_pil_image(data[i]).save(f'/home/vieira/datasets/SelfSupervisedLensless/mini_FFHQ/rest_radial_25per_prep/{basename(fname[i])}')
            # np.save(f'/home/vieira/datasets/SelfSupervisedLensless/mini_FFHQ/rest_radial_25per_prep/{basename(fname[i]).replace("png", "npy")}', data[i].numpy())
    