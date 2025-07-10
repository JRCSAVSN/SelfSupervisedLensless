import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image, center_crop, resize, pad
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser(description='Create dataset for lensless imaging')
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--psf_path', type=str, required=True, help='Path to the PSF')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the dataset')
parser.add_argument('--noise_std', type=float, default=0.01, help='Standard deviation of the noise')
args = parser.parse_args()

if __name__ == '__main__':
    # Find all images (.png, .jpg, .jpeg, .tiff, .bmp) in the data_path
    image_paths = []

    # for ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
    #     image_paths.extend(glob(os.path.join(args.data_path, f'*.{ext}')))
    # Find all images (png) in the data_path and recursively in the subdirectories
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.tiff') or file.endswith('.bmp'):
                image_paths.append(os.path.join(root, file))
    if len(image_paths) == 0:
        raise ValueError('No images found in the data_path')
    
    # Load the PSF
    psf = np.load(args.psf_path)
    psf = torch.tensor(psf, dtype=torch.float32)
    psf = psf.unsqueeze(0).unsqueeze(0)
    psf /= psf.sum()

    # Create the output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Create the dataset
    for fname in tqdm(image_paths):
        im = Image.open(fname)
        im = to_tensor(im)
        im = resize(im, psf.shape[-1]//2, interpolation=Image.BICUBIC)
        im = pad(im, (psf.shape[-1]-im.shape[-1])//2)
        
        im = im.unsqueeze(0)

        # Apply the forward model
        y = shift_invariant_model(im, psf)
        y_norm = y / y.max()
        y_norm += args.noise_std * torch.randn_like(y_norm).clip(0, 1)
        y_norm = y_norm.clamp(0, 1)
        y = y_norm * y.max()

        # plt.subplot(1, 3, 1)
        # plt.imshow(im.squeeze().numpy().transpose(1, 2, 0))
        # plt.subplot(1, 3, 2)
        # plt.imshow(psf.squeeze().numpy(), cmap='gray')
        # plt.subplot(1, 3, 3)
        # plt.imshow(y.squeeze().numpy().transpose(1, 2, 0))
        # plt.show()

        # Save the image
        np.save(args.output_path + '/' + os.path.basename(fname).split('.')[0] + '.npy', y.squeeze().numpy().transpose(1, 2, 0))

