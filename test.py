import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os.path import basename
import time

from utils.data_utils import LenslessDatasetTest, get_psf
from models.model import ReconstructionModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
args = argparser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prep_data(x):
    x -= x.min()
    x /= x.max()
    return x

if __name__ == '__main__':

    # Load configuration file
    config = load_config(args.config)
    print(config)

    # Check if save_path exists
    to_save_path = config['data']['save_path'].replace('experiments/', '')
    # to_save_path = 'unet'
    # if 'admm' in config['data']['save_path']:
    #     to_save_path += '+admm/'
    # elif 'wiener' in config['data']['save_path']:
    #     to_save_path += '+wiener/'
    # if 'simulation' in config['data']['save_path']:
    #     to_save_path = 'simulation/' + to_save_path
    # else:
    #     to_save_path = 'prototype/' + to_save_path

    if not os.path.exists(config['data']['save_path']):
        raise FileNotFoundError('Save path does not exist, experiment has not been performed yet')
    if not os.path.exists(f'reconstructions/{to_save_path}/'):
        os.makedirs(f'reconstructions/{to_save_path}/')

    # Load dataset and dataloader
    test_set = LenslessDatasetTest(data_path=config['data']['test_data_path'], data_format=config['data']['data_format'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'])

    # Load PSF
    psf = get_psf(config['deconv_layer']['psf'])
    psf = psf.to(config['training']['device'])

    # Instantiate model
    model = ReconstructionModel(
        {
            'deconv_layer': config['model']['deconv_layer'],
            'deconv_params': config['deconv_layer'],

            'encod_layer': config['model']['encod_layer'],
            'encod_params': config['encod_layer'],

            'decoder_layer': config['model']['decoder_layer'],
            'decoder_params': config['decoder_layer']
        }
    )
    model = model.to(config['training']['device'])
    state_dict = torch.load(config['data']['save_path'] + f'/model.pth')
    model.load_state_dict(state_dict)

    full_time = []

    # Training loop
    with torch.no_grad():
        test_loss = 0.0
        for data, fname in tqdm(test_loader):
            # Get the inputs
            meas = data.to(config['training']['device'])
            # Forward pass
            start_time = time.time()
            x_hat = model(meas)
            full_time.append(time.time() - start_time)
            # break
            # Save x_hat as images
            x_hat = x_hat.cpu().detach()

            for idx in range(x_hat.shape[0]):
                to_pil_image(prep_data(x_hat[idx])).save(f'reconstructions/{to_save_path}/{basename(fname[idx]).split(".")[0]}.png')
                # np.save(config['data']['save_path'] + f'/preds/{basename(fname[idx]).split(".")[0]}.npy', x_hat[idx].numpy())

                # np.save(f'reconstructions/{to_save_path}/{basename(fname[idx]).split(".")[0]}.npy', x_hat[idx].numpy())

    print('last to_save_path:', to_save_path)
    print('Testing finished')
    print('Average inference time:', sum(full_time[1:])/len(full_time[1:]))
    print()
