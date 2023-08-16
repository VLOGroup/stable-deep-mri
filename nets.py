from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import conv
import utils


class ReLU(nn.Module):
    def __init__(self, leak=0.):
        super().__init__()
        self.leak = leak

    def forward(self, x):
        where = x < 0
        return (
            th.where(where, self.leak * x, x),
            th.where(where, self.leak * th.ones_like(x), th.ones_like(x))
        )


class EnergyNet(nn.Module):
    def __init__(
        self,
        n_c: int = 1,
        n_f: int = 32,
        leak: float = 0.05,
        n_stages: int = 6,
        imsize: int = 320,
        f_mul: float = 2.,
        pot: str = 'linear',
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            conv.Conv2d(n_c, n_f, 3),
            conv.ConvScale2d(int(f_mul**0) * n_f,
                             int(f_mul**1) * n_f),
        ])
        # start from 1 since we already have the first
        for s in range(1, n_stages):
            f_in = int(f_mul**s) * n_f
            f_out = int(f_mul**(s + 1)) * n_f
            self.convs.append(conv.Conv2d(f_in, f_in))
            self.convs.append(conv.ConvScale2d(f_in, f_out))

        # compute kernel size such that last layer is essentially FC
        size_last = int(imsize * 0.5**n_stages)
        self.convs.append(
            conv.Conv2d(f_out, 1, kernel_size=size_last, pad=False)
        )

        self.act = ReLU(leak)
        self.paddings = [0, 1] * (len(self.convs) // 2) + [0]
        if pot == 'linear':
            self.__pot = lambda x: x
            self.__act = lambda x: th.ones_like(x)
        elif pot == 'abs':
            self.__pot = lambda x: th.abs(x)
            self.__act = lambda x: th.sign(x)

    def _potential(self, x):
        return self.__pot(x)

    def _activation(self, x):
        return self.__act(x)

    def _transformation(self, x):
        self.activations = [th.ones_like(x)]
        for conv_ in self.convs[:-1]:
            x, act_prime = self.act(conv_(x))
            self.activations.append(act_prime)
        return self.convs[-1](x)

    def _transformation_T(self, x):
        for act, conv_, pad in zip(
            self.activations[::-1], self.convs[::-1], self.paddings[::-1]
        ):
            x = conv_.backward(x, output_padding=pad) * act
        return x

    def forward(self, x):
        return self.energy(x)

    def energy(self, x):
        return self._potential(self._transformation(x))

    def grad(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = self._transformation(x)
        energy = self._potential(x)
        x = self._activation(x)
        grad = self._transformation_T(x)
        return energy, grad


class EnergyNetMR(EnergyNet):
    def __init__(
        self,
        n_c: int = 1,
        n_f: int = 32,
        leak: float = 0.05,
        n_stages: int = 6,
        imsize: int = 320,
        f_mul: float = 2.,
    ):
        super().__init__(n_c, n_f, leak, n_stages, imsize, f_mul, pot='abs')

    def energy(self, x):
        return super().energy(utils.rot180(utils.mri_crop(x)))

    def grad(self, x):
        inp = utils.rot180(utils.mri_crop(x))
        e, g_super = super().grad(inp)
        g_super = utils.rot180(g_super)
        pad_w = (x.shape[-1] - inp.shape[-1]) // 2
        pad_h = (x.shape[-2] - inp.shape[-2]) // 2
        g = F.pad(g_super, (pad_w, pad_w, pad_h, pad_h))
        return e, g


class CharbTV(nn.Module):
    def __init__(
        self,
        eps: float = 1e-2,
    ):
        super().__init__()
        self.nabla = utils.Grad()
        self.div = utils.Div()
        self.eps = eps

    def forward(self, x):
        return self.energy(x)

    def energy(self, x):
        nabla_u = self.nabla @ x
        norm_Du = th.sqrt((nabla_u**2).sum(dim=-1) + self.eps**2)
        e = norm_Du.sum((1, 2, 3), keepdim=True)
        return e

    def grad(self, x):
        nabla_u = self.nabla @ x
        norm_Du = th.sqrt((nabla_u**2).sum(dim=-1) + self.eps**2)
        e = norm_Du.sum((1, 2, 3), keepdim=True)
        g = self.div @ (nabla_u / norm_Du[..., None])
        return e, g
