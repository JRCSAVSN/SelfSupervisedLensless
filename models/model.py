# Class containing model and training information

import torch
import torch.nn as nn

from .physicsbased import WienerDeconv, UnrolledAdmmDeconv, AdmmDeconv
from .LenslessLearning.unet import UNet270480
from .UDN.skip import skip


deconv_opt = {
    'WienerDeconv': WienerDeconv,
    'UnrolledAdmmDeconv': UnrolledAdmmDeconv,
    'AdmmDeconv': AdmmDeconv,
    'None': None
}

encod_opt = {
    # 'Resnet': ResNetEncoder,
    'None': None
}

decoder_opt = {
    'UNet270480': UNet270480,
    'skip': skip,
    'None': None
}

class ReconstructionModel(nn.Module):
    def __init__(self, params):
        super(ReconstructionModel, self).__init__()
        self.deconv = None
        self.encoder = None
        self.decoder = None

        self.relu = nn.ReLU()

        if params['deconv_layer'] != 'None':
            self.deconv = deconv_opt[params['deconv_layer']](**params['deconv_params'])
        if params['encod_layer'] != 'None':
            self.encoder = encod_opt[params['encod_layer']](**params['encod_params'])
        if params['decoder_layer'] != 'None':
            self.decoder = decoder_opt[params['decoder_layer']](**params['decoder_params'])

    def forward(self, x):
        if self.deconv is not None:
            x = self.deconv(x)

        if self.encoder is not None:
            x = self.encoder(x)

        if self.decoder is not None:
            x = self.decoder(x)

        return self.relu(x)
