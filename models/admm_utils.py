# Original source code taken from DiffuserCam tutorial repository: https://github.com/Waller-Lab/DiffuserCam-Tutorial


###################################
###################################
# admm code
###################################
###################################

from multiprocessing import Value
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import fft
from typing import Dict

torch.autograd.set_detect_anomaly(True)

# from schema import Parameters, DATA_PATH

@dataclass
class Parameters:
    mu1: float = 1e-6
    mu2: float = 1e-5
    mu3: float = 4e-5
    tau: float = 0.0001


def SoftThresh(x: torch.Tensor, tau):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)


def Psi(v):
    return torch.stack((torch.roll(v, shifts=1, dims=2) - v, torch.roll(v, shifts=1, dims=3) - v), dim=4)
    # return np.stack((np.roll(v, 1, axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)


def M(vk: torch.Tensor, H_fft: torch.Tensor):
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk, dim=(-2, -1))) * H_fft), dim=(-2, -1)))


def PsiT(U):
    diff1 = torch.roll(U[..., 0], shifts=-1, dims=2) - U[..., 0]
    diff2 = torch.roll(U[..., 1], shifts=-1, dims=3) - U[..., 1]
    return diff1 + diff2


def uint8(frame):
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
    # frame_norm = frame/frame.max()
    return (255 * frame_norm).astype('uint8')


def MT(x, H_fft):
    x_zeroed = fft.ifftshift(x, dim=(-2, -1))
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * torch.conj(H_fft)), dim=(-2, -1)))


class UnrolledADMM(torch.nn.Module):
    def __init__(
            self,
            psf: torch.Tensor,
            # measurement: torch.Tensor,
            iters: int,
            hyper_parameter: Dict[str, float],
    ):  # -> None:
        """
        Initialize the admm algorithm
        psf and measurement must be (N,C,H,W) tensors
        :param psf:
        :param measurement:
        :param iters:
        :param hyper_parameter:
        """
        super(UnrolledADMM, self).__init__()
        self.psf = torch.nn.Parameter(psf, requires_grad=False)
        # self.measurement = measurement
        self.iters = iters
        # self.hyper_parameter: Parameters = hyper_parameter
        # self.hyper_parameter = torch.nn.Parameter(torch.Tensor([hyper_parameter['mu1'], hyper_parameter['mu2'], hyper_parameter['mu3'], hyper_parameter['tau']]).expand(iters, 4), requires_grad=True)
        self.hyper_parameter = torch.nn.Parameter(torch.ones(self.iters, 4) * torch.Tensor(
            [hyper_parameter['mu1'], hyper_parameter['mu2'], hyper_parameter['mu3'], hyper_parameter['tau']]),
                                                  requires_grad=True)
        self.sensor_size = list(psf.shape[-2:])
        self.full_size = [i * 2 for i in self.sensor_size]

    @property
    def device(self):
        return self.psf.device

    def U_update(self, eta: float, image_est: torch.Tensor, step_k: int):
        return SoftThresh(
            Psi(image_est) + eta / self.hyper_parameter[step_k, 1],
            self.hyper_parameter[step_k, 3] / self.hyper_parameter[step_k, 1]
        )

    def X_update(
            self,
            xi: torch.Tensor,
            image_est: torch.Tensor,
            H_fft: torch.Tensor,
            sensor_reading: torch.Tensor,
            X_divmat: torch.Tensor,
            step_k: int
    ):
        return X_divmat * (
                xi + self.hyper_parameter[step_k, 0] *
                M(image_est, H_fft) +
                self.CT(sensor_reading)
        )

    def C(self, m):
        # Image stored as matrix (row-column rather than x-y)
        top = (self.full_size[0] - self.sensor_size[0]) // 2
        bottom = (self.full_size[0] + self.sensor_size[0]) // 2
        left = (self.full_size[1] - self.sensor_size[1]) // 2
        right = (self.full_size[1] + self.sensor_size[1]) // 2
        return m[:, :, top:bottom, left:right]

    def CT(self, b):
        v_pad = (self.full_size[0] - self.sensor_size[0]) // 2
        h_pad = (self.full_size[1] - self.sensor_size[1]) // 2
        return torch.nn.functional.pad(b, (h_pad, h_pad, v_pad, v_pad), mode='constant', value=0)

    def precompute_X_divmat(self, step_k: int):
        """Only call this function once!
        Store it in a variable and only use that variable
        during every update step"""
        return 1. / (self.CT(torch.ones([1, 1] + self.sensor_size, device=self.device)) + self.hyper_parameter[
            step_k, 0])

    def W_update(self, rho: torch.Tensor, image_est: torch.Tensor, step_k):
        return torch.clamp(rho / self.hyper_parameter[step_k, 2] + image_est, min=0)

    def r_calc(self, w, rho, u, eta, x, xi, H_fft, step_k):
        return (self.hyper_parameter[step_k, 2] * w - rho) + \
            PsiT(self.hyper_parameter[step_k, 1] * u - eta) + \
            MT(self.hyper_parameter[step_k, 0] * x - xi, H_fft)

    def V_update(self, w, rho, u, eta, x, xi, H_fft, R_divmat, step_k):
        freq_space_result = R_divmat * fft.fft2(
            fft.ifftshift(self.r_calc(w, rho, u, eta, x, xi, H_fft, step_k), dim=(-2, -1)))
        return torch.real(fft.fftshift(fft.ifft2(freq_space_result), dim=(-2, -1)))

    def precompute_PsiTPsi(self):
        PsiTPsi = torch.zeros([1, 1] + self.full_size, device=self.device)
        PsiTPsi[:, :, 0, 0] = 4
        PsiTPsi[:, :, 0, 1] = PsiTPsi[:, :, 1, 0] = PsiTPsi[:, :, 0, -1] = PsiTPsi[:, :, -1, 0] = -1
        PsiTPsi = fft.fft2(PsiTPsi)
        return PsiTPsi

    def precompute_R_divmat(self, H_fft: torch.Tensor, PsiTPsi: torch.Tensor, step_k: int):
        """Only call this function once!
        Store it in a variable and only use that variable
        during every update step"""
        MTM_component = self.hyper_parameter[step_k, 0] * (torch.abs(torch.conj(H_fft) * H_fft))
        PsiTPsi_component = self.hyper_parameter[step_k, 1] * torch.abs(PsiTPsi)
        id_component = self.hyper_parameter[step_k, 2]
        """This matrix is a mask in frequency space. So we will only use
        it on images that have already been transformed via an fft"""
        return 1. / (MTM_component + PsiTPsi_component + id_component)

    def xi_update(self, xi, V, H_fft, X, step_k):
        return xi + self.hyper_parameter[step_k, 0] * (M(V, H_fft) - X)

    def eta_update(self, eta, V, U, step_k):
        return eta + self.hyper_parameter[step_k, 1] * (Psi(V) - U)

    def rho_update(self, rho: torch.Tensor, V: torch.Tensor, W: torch.Tensor, step_k: int):
        return rho + self.hyper_parameter[step_k, 2] * (V - W)

    def init_Matrices(self, H_fft: torch.Tensor):
        X = torch.zeros([1, 1] + self.full_size, device=self.device)
        U = torch.zeros([1, 1] + [self.full_size[0], self.full_size[1], 2], device=self.device)
        V = torch.zeros([1, 1] + self.full_size, device=self.device)
        W = torch.zeros([1, 1] + self.full_size, device=self.device)

        xi = torch.zeros_like(M(V, H_fft), device=self.device)
        eta = torch.zeros_like(Psi(V), device=self.device)
        rho = torch.zeros_like(W, device=self.device)
        return X, U, V, W, xi, eta, rho

    def precompute_H_fft(self, psf: torch.Tensor):
        return fft.fft2(fft.ifftshift(self.CT(psf), dim=(-2, -1)))

    def ADMM_Step(self, X, U, V, W, xi, eta, rho, precomputed, step_k):
        H_fft, data, X_divmat, R_divmat = precomputed
        U = self.U_update(eta, V, step_k)
        X = self.X_update(xi, V, H_fft, data, X_divmat, step_k)
        V = self.V_update(W, rho, U, eta, X, xi, H_fft, R_divmat, step_k)
        W = self.W_update(rho, V, step_k)
        xi = self.xi_update(xi, V, H_fft, X, step_k)
        eta = self.eta_update(eta, V, U, step_k)
        rho = self.rho_update(rho, V, W, step_k)

        return X, U, V, W, xi, eta, rho

    def run(self, measurement: torch.Tensor):
        H_fft = self.precompute_H_fft(self.psf)
        X, U, V, W, xi, eta, rho = self.init_Matrices(H_fft)
        X_divmat = self.precompute_X_divmat(0)
        PsiTPsi = self.precompute_PsiTPsi()
        R_divmat = self.precompute_R_divmat(H_fft, PsiTPsi, 0)

        for step_k in range(self.iters):
            X, U, V, W, xi, eta, rho = self.ADMM_Step(X, U, V, W, xi, eta, rho,
                                                      [H_fft, measurement, X_divmat, R_divmat], step_k)

            X_divmat = self.precompute_X_divmat(step_k)
            PsiTPsi = self.precompute_PsiTPsi()
            R_divmat = self.precompute_R_divmat(H_fft, PsiTPsi, step_k)

        return self.C(V)