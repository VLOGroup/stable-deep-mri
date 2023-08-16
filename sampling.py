import math
from typing import Callable

import torch as th


def bdls(
    xi: th.Tensor,
    R: th.nn.Module,
    *,
    K: Callable[[th.Tensor], th.Tensor],
    dt: float = 1e-3,
    n: int = 500,
    lamda: float = 1e3,
):
    '''
    Birth-Death accelerated Langevin
    https://arxiv.org/pdf/1905.09863.pdf (Algorithm 1)
    '''
    x = xi.clone()
    sqrt_2e = math.sqrt(2 * dt)
    for i in range(n):
        f, g = R.grad(x)
        x -= dt * g + th.randn_like(x) * sqrt_2e
        beta = f.squeeze() + K(x)
        beta_bar = beta - beta.mean(0, keepdim=True)
        really_dup = (1 - th.exp(beta_bar * dt)) > th.rand_like(beta)
        really_kill = (1 - th.exp(-beta_bar * dt)) > th.rand_like(beta)
        x[really_kill & (beta_bar > 0)] = x[~really_kill][th.randperm(
            really_kill.sum()
        )].clone()
        x[~really_dup][th.randperm(really_dup.sum())
                       ] = x[really_dup & (beta_bar < 0)].clone()
    return x


def ula(
    x_init: th.Tensor,
    nabla: Callable[[th.Tensor], th.Tensor],
    n: int = 500,
    epsilon: float = 7.5e-3,
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
) -> th.Tensor:
    x = x_init.clone()
    for i in range(n):
        x -= nabla(x) + epsilon * th.randn_like(x)
        callback(x, i)

    return x


def proximal_ula(
    f_init: th.Tensor,
    nabla: Callable[[th.Tensor], th.Tensor],
    prox: Callable[[th.Tensor], th.Tensor],
    n: int = 200,
    epsilon: float = 7.5e-3,
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
) -> th.Tensor:
    '''
    Implements the proximal unadjusted Langevin algorithm
    https://arxiv.org/pdf/1306.0187.pdf
    '''
    f = f_init.clone()
    for i in range(n):
        f = prox(f - nabla(f)) + epsilon * th.randn_like(f)
        callback(f, i)
    return f


def myula(
    f_init: th.Tensor,
    nabla: Callable[[th.Tensor], th.Tensor],
    prox: Callable[[th.Tensor, float], th.Tensor],
    gamma: float = 0.004,
    lamda: float = 0.01,
    n: int = 200,
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
):
    '''
    we divide the whole thing by gamma to ease implementation and scaling of
    noise and nabla f
    '''
    f = f_init.clone()
    for i in range(n):
        f = (1 - gamma / lamda) * f - gamma * nabla(f) + \
            gamma / lamda * prox(f, lamda) + gamma * th.randn_like(f) * 7.5e-3
        callback(f, i)
    return f


def pula(
    x: th.Tensor,
    grad_fn: th.nn.Module,
    *,
    epsilon: float = 7.5e-3,
    n: int = 500,
    beta: float = 0.99,
    lamda: float = 7e-1,
    callback=lambda x, i: None,
):
    '''
    preconditioned ULA
    https://arxiv.org/pdf/1512.07666.pdf
    '''
    V = th.zeros_like(x)
    for i in range(n):
        g = grad_fn(x)
        V = beta * V + (g**2) * (1 - beta)
        G = V.sqrt() + lamda
        x = x - (g + epsilon * th.randn_like(x)) / G
        callback(x, i)

    return x


def mala(
    x_init,
    e_g,
    *,
    n=1,
    a=1,
    beta=1,
    verbose=False,
    check=1,
    writer=None,
    set='train',
    step=0,
    callback=None,
):
    x = x_init.clone()
    red_dims = tuple(range(1, x.ndim))

    # initial step size
    tmp = (1, ) * (x.ndim - 1)
    L = x.new_ones(x.shape[0], *tmp)
    tmp = th.ones_like(L)

    # compute the initial gradient and energy
    energy_x, grad_x = e_g(x)

    for it in range(n):
        # compute the candidate
        z = x - grad_x / L + th.sqrt(2 / L) * th.randn_like(x) * beta

        # compute the acceptance probability
        energy_z, grad_z = e_g(z)
        p_x = -th.sum(energy_x, red_dims, keepdims=True) - th.sum(
            (z - x + grad_x / L)**2, red_dims, keepdims=True
        ) / 4 * L
        p_z = -th.sum(energy_z, red_dims, keepdims=True) - th.sum(
            (x - z + grad_z / L)**2, red_dims, keepdims=True
        ) / 4 * L
        alpha = th.exp(p_z - p_x).clip_(0, 1)

        # accept with probability alpha
        u = th.rand_like(alpha)
        accept = (u < alpha).float()
        x.copy_(z * accept + x * (1 - accept))
        energy_x.copy_(energy_z * accept + energy_x * (1 - accept))
        grad_x.copy_(grad_z * accept + grad_x * (1 - accept))

        # adjust step size
        L *= th.where(alpha < 0.574, tmp * 1.1, tmp / 1.1)
        # L.clamp_(min=0.5)
        callback(x, it)

        if verbose and it % check == 0:
            print(f'{it:04d}', 'alpha:', alpha.cpu().numpy().squeeze())
            print(f'{it:04d}', 'accept:', accept.cpu().numpy().squeeze())
            print(f'{it:04d}', 'L:', L.cpu().numpy().squeeze())

    x_avg = th.zeros_like(x)

    for it in range(a):
        x_avg += x / a
        # compute the candidate
        z = x - grad_x / L + th.sqrt(2 / L) * th.randn_like(x) * beta

        # compute the acceptance probability
        energy_z, grad_z = e_g(z)
        p_x = -th.sum(energy_x, red_dims, keepdims=True) - th.sum(
            (z - x + grad_x / L)**2, red_dims, keepdims=True
        ) / 4 * L
        p_z = -th.sum(energy_z, red_dims, keepdims=True) - th.sum(
            (x - z + grad_z / L)**2, red_dims, keepdims=True
        ) / 4 * L
        alpha = th.exp(p_z - p_x).clip_(0, 1)

        # accept with probability alpha
        u = th.rand_like(alpha)
        accept = (u < alpha).float()
        x.copy_(z * accept + x * (1 - accept))
        energy_x.copy_(energy_z * accept + energy_x * (1 - accept))
        grad_x.copy_(grad_z * accept + grad_x * (1 - accept))

        # adjust step size
        L *= th.where(alpha < 0.574, tmp * 2, tmp / 1.5)
        # L.clamp_(min=5e3)

        if verbose and it % check == 0:
            print(it, 'alpha', alpha.cpu().numpy().squeeze())
            print(it, 'accept', accept.cpu().numpy().squeeze())
            print(it, 'L', L.cpu().numpy().squeeze())

    return x_avg, L
