import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import os



from utils.data_utils import LenslessDataset, RandomSubsetSampler, get_psf
from utils.lensless_utils import shift_invariant_model
from models.model import ReconstructionModel

argparser = argparse.ArgumentParser()

argparser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
# # Parse args
args = argparser.parse_args()

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':

    # Load configuration file
    config = load_config(args.config)
    print(config)

    # Set up tensorboard writer
    writer = SummaryWriter(config['data']['save_path'] + '/logs/')

    # Check if save_path exists
    if not os.path.exists(config['data']['save_path']):
        os.makedirs(config['data']['save_path'])


    # Load dataset and dataloader
    train_set = LenslessDataset(data_path=config['data']['train_data_path'], data_format= config['data']['data_format'])
    if config['training']['sampler_size'] != 'None':
        print('Creating Sampler')
        sampler = RandomSubsetSampler(train_set, config['training']['sampler_size'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['training']['batch_size'], sampler=sampler, num_workers=config['training']['num_workers'])
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True)

    test_set = LenslessDataset(data_path=config['data']['test_data_path'], data_format=config['data']['data_format'])
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

    criterion = torch.nn.MSELoss()

    optim_setting = []
    if config['deconv_layer'] != "None" and config['training']['deconv_lr'] != "None":
        optim_setting.append({'params': model.deconv.parameters(), 'lr': config['training']['deconv_lr']})
    if config['encod_layer'] != "None" and config['training']['encoder_lr'] != "None":
        optim_setting.append({'params': model.encoder.parameters(), 'lr': config['training']['encoder_lr']})
    if config['decoder_layer'] != "None" and config['training']['decoder_lr'] != "None":
        optim_setting.append({'params': model.decoder.parameters(), 'lr': config['training']['decoder_lr']})
    optimizer = torch.optim.Adam(optim_setting)
    
    # Training loop
    for epoch in tqdm(range(config['training']['epochs'])):
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            meas = data.to(config['training']['device'])
            optimizer.zero_grad()
            x_hat = model(meas)
            meas_hat = shift_invariant_model(x_hat, psf)
            loss = criterion(meas_hat, meas)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar('training_loss (iter)', loss.item(), epoch*len(train_loader) + i)
        train_loss /= len(train_loader)
        writer.add_scalar('training_loss (epoch)', train_loss, epoch)

        if epoch % 10 == 0 or epoch == config['training']['epochs'] - 1:
            with torch.no_grad():
                test_loss = 0.0
                for i, data in enumerate(test_loader):
                    meas = data.to(config['training']['device'])
                    x_hat = model(meas)
                    meas_hat = shift_invariant_model(x_hat, psf)
                    loss = criterion(meas_hat, meas)
                    test_loss += loss.item()
                test_loss /= len(test_loader)
            writer.add_scalar('test_loss', test_loss, epoch)

            # Save model
            torch.save(model.state_dict(), config['data']['save_path'] + f'/model_{epoch}.pth')

    torch.save(model.state_dict(), config['data']['save_path'] + f'/model.pth')
    writer.close()
    print('Finished Training')





