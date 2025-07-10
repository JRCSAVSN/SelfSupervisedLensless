import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import to_tensor, to_pil_image, pad, resize, center_crop
from os.path import basename

OG_DSET_PATH = 'home/vieira/datasets/FFHQ/'

class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size
    
    def __iter__(self):
        idx = torch.randperm(len(self.data_source))[:self.subset_size]
        return iter(idx.tolist())
    
    def __len__(self):
        return self.subset_size
    
class NonRandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size
    
    def __iter__(self):
        idx = torch.arange(self.subset_size)
        return iter(idx.tolist())
    
    def __len__(self):
        return self.subset_size

def get_psf(path):
    p = np.load(path)
    p = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float()
    p /= p.sum()
    return p
    
class LenslessDataset(Dataset):
    def __init__(self, data_path, data_format='png'):
        self.data_path = data_path
        # select all png and tiff files

        self.fnames_1 = glob(self.data_path + f'/*.{data_format}', recursive=True)
        # self.fnames_1.sort(key=lambda x: int(basename(x).split('.')[0]))
        self.data_format = data_format

        # assert len(self.fnames_1) > 0, f'Dataset should have more than 0 {data_format} images'
        assert len(self.fnames_1) > 0, f'Dataset should have more than 0 {data_format} images, found {len(self.fnames_1)} instead'

    def __len__(self):
        return len(self.fnames_1)
        
    def __getitem__(self, idx):
        if self.data_format == 'npy':
            img_1 = np.load(self.fnames_1[idx])
        else:
            img_1 = Image.open(self.fnames_1[idx])
        
        img_1 = to_tensor(img_1)
        # img_1 -= img_1.min()
        # img_1 /= img_1.max()

        return img_1


class LenslessDatasetTest(Dataset):
    def __init__(self, data_path, data_format='png'):
        self.data_path = data_path
        self.fnames_1 = glob(self.data_path + f'/*.{data_format}', recursive=True)
        self.fnames_1.sort(key=lambda x: int(basename(x).split('.')[0]))
        self.data_format = data_format

        assert len(self.fnames_1) > 0, f'Dataset should have more than 0 {data_format} images'

    def __len__(self):
        return len(self.fnames_1)

    def __getitem__(self, idx):
        if self.data_format == 'npy':
            img_1 = np.load(self.fnames_1[idx])
        else:
            img_1 = Image.open(self.fnames_1[idx])

        img_1 = to_tensor(img_1)
        # img_1 -= img_1.min()
        # img_1 /= img_1.max()

        return img_1, self.fnames_1[idx]


class LenslessDatasetTestV2(Dataset):
    def __init__(self, data_path, data_format='png'):
        self.data_path = data_path
        self.fnames_1 = glob(self.data_path + f'/*.{data_format}', recursive=True)
        # self.fnames_1.sort(key=lambda x: int(basename(x).split('.')[0]))
        self.data_format = data_format

        assert len(self.fnames_1) > 0, f'Dataset should have more than 0 {data_format} images'

    def __len__(self):
        return len(self.fnames_1)

    def __getitem__(self, idx):
        if self.data_format == 'npy':
            img_1 = np.load(self.fnames_1[idx])
        else:
            img_1 = Image.open(self.fnames_1[idx])

        img_1 = to_tensor(img_1)
        # img_1 -= img_1.min()
        # img_1 /= img_1.max()

        gt = None
        gt = Image.open(OG_DSET_PATH + f'{(idx // 1000) * 1000}/{self.fnames_1[idx].split("/")[-1].split(".")[0]}.png')
        gt = to_tensor(gt)
        gt = resize(gt, 128, interpolation=Image.BICUBIC)
        gt = gt.clip(0, 1)

        return img_1, gt, self.fnames_1[idx]