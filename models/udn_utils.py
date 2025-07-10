from multiprocessing import Value

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, resize, center_crop, pad, crop
from tqdm import tqdm

from .UDN.Forward import Forward_Model
from .UDN.skip import skip
from .UDN.util import uint8
from .UDN.schema import Parameters, DATA_PATH


class UDN:
    def __init__(
            self,
            psf: torch.Tensor,
            measurement: torch.Tensor,
            iters: int,
            batch_size: int = 1,
            lr = 1e-2,
            device='cpu',
            eval_iter=1000,
            pretrained_model=None,
            pretrained_model_input=None,
    ):
        """
        Initialize the udn algorithm
        psf and measurement must be (1,C,H,W) tensors
        :param psf:
        :param measurement:
        :param iters:
        """
        self.device = device
        self.psf = psf.float()
        self.sensor_size = list(psf.shape[-2:])
        self.iters = iters
        self.eval_iter = eval_iter
        self.batch_size = batch_size
        self.pretrained_model = pretrained_model
        self.pretrained_model_input = pretrained_model_input

        assert self.pretrained_model is None or self.pretrained_model_input is not None, "If pretrained model is given, pretrained model input must be given"

        if pretrained_model:
            self.net = pretrained_model.to(device)
            self.net_input = pretrained_model_input.to(device)
        else:
            self.net = skip(
                3,
                3,
                num_channels_down=[128] * 5,
                num_channels_up=[128] * 5,
                num_channels_skip=[4] * 5,
                upsample_mode='bilinear',
                downsample_mode='stride',
                need_sigmoid=False,
                need_bias=True,
                pad='reflection',
                act_fun='LeakyReLU',
            ).to(device)
            self.net_input = torch.zeros([1, 3] + self.sensor_size).uniform_().to(device) * 0.1  # b,3, h, w
        # if pretrained_model_path:
        #     self.net.load_state_dict(torch.load(pretrained_model_path))
        self.forward_model = Forward_Model(self.psf, cuda_device=device)
        # self.measurement = torch.cat([measurement.transpose(1,0) for _ in range(batch_size)], dim=0)  # b, 3, h, w
        self.measurement = measurement
        self.measurement = self.measurement.float()
        
        self.noise = torch.zeros([batch_size, 3] + self.sensor_size).to(device)  # b,3 ,h,w
        self.reg_noise = 1/30

        # Losses
        self.mse = torch.nn.MSELoss().to(device)

        # optimizer definition
        p = self.net.parameters()
        self.optimizer = torch.optim.Adam(p, lr=lr)

    def run_udn(self):
        reconstruction = None
        recon_hist = []

        for i in tqdm(range(self.iters)):
            self.optimizer.zero_grad()

            input_matrix = self.net_input + (self.noise.normal_() * self.reg_noise)

            reconstruction = self.net(input_matrix)
            gen_meas = self.forward_model.forward_zero_pad(reconstruction)
            # gen_meas = torch.nn.functional.normalize(gen_meas, dim=[1, 2, 3], p=2) # if PSF is not normalized(sum!=1.0), you may need
            # this part.
            loss = self.mse(gen_meas, self.measurement)
            # loss += tv_weight * df.tv_loss(reconstruction)

            loss.backward()
            self.optimizer.step()

            if i % self.eval_iter == 0:
                r = reconstruction[0].detach().cpu().permute(1, 2, 0).numpy()
                r -= r.min()
                r /= r.max()
                # to_pil_image(r).save(f'workspace/recon_{i}.png')

                recon_hist.append(r)

        return np.array(recon_hist)
