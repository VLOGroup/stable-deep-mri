import io
import os
import json
import math
import re
from typing import Tuple, Union

import fastmri.models as fm
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from skimage.morphology import skeletonize
from skimage.transform import rotate

import cfl
import nets


def grid_search(grid, callback):
    best = float('-inf')
    arg_best = 0
    for p in grid:
        value = callback(p)
        if value > best:
            best = value
            arg_best = p

    return best, arg_best


class ProxH1(th.nn.Module):
    def __init__(self, H: int, W: int, mu: float) -> None:
        super().__init__()
        wy = (2 - 2 * th.cos(np.pi * th.arange(H) / H))
        wx = (2 - 2 * th.cos(np.pi * th.arange(W) / W))
        Wx, Wy = th.meshgrid(wx, wy, indexing='xy')
        self.register_buffer('W', Wx + Wy)
        self.W: th.Tensor  # For type hinting
        self.mu = mu

    def forward(self, x: th.Tensor, tau: float) -> th.Tensor:
        gamma = 1 / (tau * self.mu)
        c_real, c_imag = th.view_as_real(x).permute(4, 0, 1, 2, 3)
        return idst2(dst2(c_real * gamma) / (self.W + gamma)) + \
            idst2(dst2(c_imag * gamma) / (self.W + gamma)) * 1j


def varnet_model():
    R = fm.VarNet().cuda()
    state_dict = th.load('./models/varnet.ckpt')
    R.load_state_dict(state_dict)
    return R.cuda()


def unet_model():
    R = fm.Unet(
        in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0
    )
    state_dict = th.load('./models/unet.ckpt')
    R.load_state_dict(state_dict)
    R = R.eval()
    return R.cuda()


def call_unet(net, samples):
    normalized, mean, std = unet_normalize(samples)
    out = th.stack([net(im[None])[0] for im in normalized])
    return out * std + mean


def mr_model():
    config_path = './config.json'
    with open(config_path) as file:
        config = json.load(file)
    imsize = 320
    R = nets.EnergyNetMR(
        f_mul=config["f_mul"],
        n_c=config["im_ch"],
        n_f=config["n_f"],
        imsize=imsize,
        n_stages=config["stages"],
    )
    R.load_state_dict(th.load('./models/ebm.ckpt'))
    return R.cuda()


def data_l2(lamda, A, p, dims=(1, 2, 3)):
    def h(u):
        return lamda / 2 * (th.abs(A @ u - p)**2).sum(dims)

    def nabla_h(u):
        return lamda * (A.H @ (A @ u - p))

    return h, nabla_h


def power_method(A, f, iter=100):
    x = A.H @ f
    s = 0
    for _ in range(iter):
        x = (A.H @ (A @ x))
        s = math.sqrt(th.sum(th.abs(x)**2).item())
        x /= s
    return s


def inner(x, y, keepdim=False):
    if th.is_complex(x) or th.is_complex(y):
        prod = (
            th.view_as_real(x.to(th.complex64)) *
            th.view_as_real(y.to(th.complex64))
        ).sum((1, 2, 3, 4))
        return prod[:, None, None, None] if keepdim else prod
    else:
        return (x * y).sum((1, 2, 3), keepdim=keepdim)


def natural_sort(li):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(li, key=alphanum_key)


def radial_mask(
    shape,
    num_spokes,
    theta=np.pi * (3 - np.sqrt(5)),
    offset=0,
    theta0=0,
    skinny=True,
    extend=True
) -> th.Tensor:
    if extend:
        mode = 'wrap'
    else:
        mode = 'constant'

    idx = np.zeros(shape, dtype=bool)
    idx0 = np.zeros(idx.shape, dtype=bool)
    idx0[int(shape[0] / 2), :] = 1

    for ii in range(num_spokes):
        idx1 = rotate(
            idx0,
            np.rad2deg(theta * (ii + offset) + theta0),
            resize=False,
            mode=mode
        ).astype(bool)
        if skinny:
            idx1 = skeletonize(idx1)
        idx |= idx1

    return th.from_numpy(idx).cuda().bool()[None, None]


def unet_normalize(
    data: th.Tensor,
    eps: Union[float, th.Tensor] = 1e-11
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    mean = data.mean(dim=(1, 2, 3), keepdim=True)
    std = data.std(dim=(1, 2, 3), keepdim=True)
    normalized = (data - mean) / (std + eps)
    return normalized, mean, std


def _mask_to_normalized_coordinates(m: np.ndarray, ) -> np.ndarray:
    # Offset such that pixels are "centered" in normalized coordinates
    to = np.pi - (np.pi / m.shape[-1])
    return shift_interval(np.argwhere(m).T, (0, m.shape[-1] - 1), (-to, to))


def _normalized_corrdinates_to_mask(
    m: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    c = m / (2 * np.pi)
    c += 0.5
    c *= shape[0]
    c = np.floor(c).astype(np.int32)
    mask = np.zeros(shape)
    mask[c[0], c[1]] = 1
    return mask


def gaussian2d_mask(
    imsize: Tuple[int, int] = (320, 320),
    acc: float = 8.,
) -> th.Tensor:
    pixels = imsize[0] * imsize[1]
    n_samples = int(pixels // acc)
    mask = np.zeros(imsize)
    cov_factor = min(imsize) * (2 / 128)
    mean = [imsize[0] // 2, imsize[1] // 2]
    cov = np.array([[imsize[0] * cov_factor, 0], [0, imsize[1] * cov_factor]])
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    int_samples = samples.astype(int)
    int_samples = np.stack([
        np.clip(int_samples[:, 0], 0, imsize[0] - 1),
        np.clip(int_samples[:, 1], 0, imsize[1] - 1),
    ])
    mask[int_samples[0], int_samples[1]] = 1
    while mask.size / mask.sum() > acc:
        samples = np.random.multivariate_normal(mean, cov, 5)
        int_samples = samples.astype(int)
        int_samples = np.stack([
            np.clip(int_samples[:, 0], 0, imsize[0] - 1),
            np.clip(int_samples[:, 1], 0, imsize[1] - 1),
        ])
        mask[int_samples[0], int_samples[1]] = 1
    return th.from_numpy(mask)[None, None].cuda().bool()


def random_mask(
    imsize: Tuple[int, int],
    acceleration: float = 4.,
):
    rng = np.random.default_rng(42)
    p_zero = 1 - 1 / acceleration
    m = rng.choice([0, 1], size=imsize, p=[p_zero, 1 - p_zero])
    return th.from_numpy(m)


def cartesian_mask(
    imsize: Tuple[int, int] = (640, 372),
    center_fraction: float = 0.08,
    acceleration: int = 4,
    vertical=True
) -> th.Tensor:
    size_random = imsize[1] if vertical else imsize[0]
    rng = np.random.default_rng(42)
    num_low_freqs = int(round(size_random * center_fraction))
    prob = (size_random / acceleration - num_low_freqs) \
        / (size_random - num_low_freqs)
    mask = rng.uniform(size=size_random) < prob
    pad = (size_random - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True
    mask = np.tile(mask, (imsize[0] if vertical else imsize[1], 1))
    if not vertical:
        mask = mask.T
    return th.from_numpy(mask)[None, None].cuda().bool()


def spiral_mask(
    power: float = 3.,
    n_points: int = 5000,
    trips: int = 10,
) -> th.Tensor:
    magn = th.linspace(0, np.sqrt(2) * np.pi**(1 / power), n_points)**power
    phase = th.linspace(0, trips * 2 * np.pi, n_points)
    traj_x, traj_y = th.view_as_real(magn * th.exp(1j * phase)).permute(1, 0)
    i_x = th.abs(traj_x) < np.pi
    i_y = th.abs(traj_y) < np.pi
    traj_x = traj_x[i_x & i_y]
    traj_y = traj_y[i_x & i_y]
    return th.from_numpy(
        _normalized_corrdinates_to_mask(
            th.stack((
                traj_y,
                traj_x,
            ), dim=0).cpu().numpy(), (320, 320)
        )
    ).cuda()


def muckley_radial(spokelength):
    nspokes = 405

    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    return th.from_numpy(np.stack((ky.flatten(), kx.flatten()), axis=0))


def plot_mask(
    mask: th.Tensor,
    path: str,
):
    m = mask.cpu().numpy().squeeze()
    sc = np.argwhere(m)
    ar = m.shape[1] / m.shape[0]
    fig, ax = plt.subplots(figsize=(15 * ar, 15))
    ax.scatter(sc[:, 1], sc[:, 0], c=[[0, 0, 0]], s=2, marker='s')
    ax.plot(
        [-0.5, m.shape[1] - 0.5, m.shape[1] - 0.5, -0.5, -0.5],
        [-0.5, -0.5, m.shape[0] + 0.5, m.shape[0] + 0.5, -0.5],
        'k',
        linewidth=2,
    )
    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def psnr(
    x: th.Tensor,
    y: th.Tensor,
    value_range: Union[th.Tensor, float] = 1.
) -> th.Tensor:
    return (10 * th.log10(value_range**2 / mse(x, y)))


def nmse(
    x: th.Tensor,
    y: th.Tensor,
) -> th.Tensor:
    return th.sum((x - y)**2, dim=(1, 2, 3),
                  keepdim=True) / th.sum(x**2, dim=(1, 2, 3), keepdim=True)


def mse(
    x: th.Tensor,
    y: th.Tensor,
) -> th.Tensor:
    return th.mean((x - y)**2, dim=(1, 2, 3), keepdim=True)


def rot180(x):
    return th.rot90(x, dims=(-2, -1), k=2)


def rss(x):
    return (th.abs(x)**2).sum(1, keepdim=True).sqrt()


def shift_interval(
    data: Union[np.ndarray, th.Tensor], from_: Tuple[float, float],
    to_: Tuple[float, float]
) -> Union[np.ndarray, th.Tensor]:
    return ((to_[1] - to_[0]) /
            (from_[1] - from_[0])) * (data - from_[0]) + to_[0]


def _get_gauss_kernel() -> th.Tensor:
    return th.tensor([[
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]]) / 256.0


def _compute_padding(kernel_size: list[int]) -> list[int]:
    computed = [k // 2 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def cifft2(z: th.Tensor) -> th.Tensor:
    return th.fft.ifftshift(
        th.fft.ifft2(th.fft.fftshift(z, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )


def cfft2(x: th.Tensor) -> th.Tensor:
    return th.fft.fftshift(
        th.fft.fft2(th.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )


def dct(x_: th.Tensor) -> th.Tensor:
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, sh[-1])
    temp = th.hstack((x[:, ::2], th.flip(x, (-1, ))[:, N % 2::2]))
    temp = th.fft.fft(temp)
    k = th.exp(-1j * np.pi * th.arange(N).to(x_) / (2 * N))
    return (temp * k).real.view(sh) / math.sqrt(N / 2)


def dct2(x: th.Tensor) -> th.Tensor:
    return dct(dct(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def idct(x_: th.Tensor) -> th.Tensor:
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, N)
    factor = -1j * np.pi / (N * 2)
    temp = x * th.exp(th.arange(N).to(x_) * factor)[None]
    temp[:, 0] /= 2
    temp = th.fft.fft(temp).real

    result = th.empty_like(x)
    result[:, ::2] = temp[:, :(N + 1) // 2]
    indices = th.arange(-1 - N % 2, -N, -2)
    result[:, indices] = temp[:, (N + 1) // 2:]
    return result.view(sh) / math.sqrt(N / 2)


def idct2(x: th.Tensor) -> th.Tensor:
    return idct(idct(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def dst(x):
    """
    https://arxiv.org/pdf/cs/0703150.pdf
    """
    m = th.ones((
        1,
        1,
        1,
        x.shape[-1],
    ), device=x.device)
    m[..., 1::2] = -1
    return dct(x * m).flip((-1, ))


def dst2(x):
    return dst(dst(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def idst(y):
    m = th.ones((
        1,
        1,
        1,
        y.shape[-1],
    ), device=y.device)
    m[..., 1::2] = -1
    return m * idct(y.flip((-1, )))


def idst2(y):
    """
    https://arxiv.org/pdf/cs/0703150.pdf
    """
    return idst(idst(y).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def view_as_real_batch(x):
    return th.cat([x.real, x.imag], 1)


def view_as_real_batch_T(x):
    n_ch = x.shape[1] // 2
    return x[:, :n_ch] + 1j * x[:, n_ch:]


def plot_to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def mri_crop(x):
    return tvf.center_crop(x, [320, 320])


def mri_crop_T(x, shape):
    pad_w = (shape[-1] - 320) // 2
    pad_h = (shape[-2] - 320) // 2
    return tf.pad(x, (pad_w, pad_w, pad_h, pad_h))


def espirit(kspace, radius):
    kspace = kspace.cpu().numpy().transpose(0, 2, 3, 1)
    cfl.writecfl('kspace.tmp', kspace)
    os.system(f"bart ecalib -c0 -m1 -r{radius} kspace.tmp sens.tmp")
    sens = cfl.readcfl('sens.tmp')
    os.system("rm kspace.tmp.cfl kspace.tmp.hdr sens.tmp.cfl sens.tmp.hdr")
    return th.from_numpy(sens.transpose(0, 3, 1, 2)).cuda()


class SSIM(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            "w",
            th.ones(1, 1, win_size, win_size) / win_size**2
        )
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
        self.w: th.Tensor

    def forward(
        self,
        X: th.Tensor,
        Y: th.Tensor,
        data_range: Union[th.Tensor, float] = 1.,
        reduced: bool = True,
    ):
        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2
        ux = tf.conv2d(X, self.w)
        uy = tf.conv2d(Y, self.w)
        uxx = tf.conv2d(X * X, self.w)
        uyy = tf.conv2d(Y * Y, self.w)
        uxy = tf.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return S.mean((1, 2, 3))
        else:
            return S


class Grad(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        grad = x.new_zeros((*x.shape, 2))
        grad[:, :, :, :-1, 0] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, :, :-1, :, 1] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def forward(self, x):
        return self @ x


class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        div = x.new_zeros(x.shape[:-1])
        div[:, :, :, 1:] += x[:, :, :, :-1, 0]
        div[:, :, :, :-1] -= x[:, :, :, :-1, 0]
        div[:, :, 1:, :] += x[:, :, :-1, :, 1]
        div[:, :, :-1, :] -= x[:, :, :-1, :, 1]
        return div

    def forward(self, x):
        return self @ x


class AsMatrix():
    def __init__(
        self,
        operator,
        adjoint,
    ):
        self.operator = operator
        self.adjoint = adjoint

    def __matmul__(
        self,
        x,
    ):
        return self.operator(x)

    def forward(self, x):
        return self @ x

    def __call__(self, x):
        return self @ x

    @property
    def H(self, ):
        return AsMatrix(self.adjoint, self.operator)
