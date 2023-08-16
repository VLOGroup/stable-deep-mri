import math
from typing import Callable, Tuple

import torch as th
from torch.optim.optimizer import Optimizer

import utils


class AdaBelief(Optimizer):
    r"""Implements a simplified version AdaBelief with projection.

    It has been proposed in `AdaBelief Optimizer: fast as Adam, generalizes as
    goodas SGD, and sufficiently stable to train GANs.`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0,
        amsgrad=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = th.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = th.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad.
                        state['max_exp_avg_sq'] = th.zeros_like(p.data)

                    state['old_weights'] = p.data.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                # Transform the gradient for the second moment
                if hasattr(p, 'reduction_dim'):
                    grad_reduced = th.sum(
                        grad_residual**2, p.reduction_dim, True
                    )
                else:
                    grad_reduced = grad_residual**2
                exp_avg_sq.mul_(beta2).add_(grad_reduced, alpha=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg.
                    th.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    #denom = exp_avg_sq.sqrt().add_(group['eps'])
                    denom = (
                        exp_avg_sq.add_(group['eps']).sqrt() /
                        math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # perform the gradient step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # perform a projection
                if hasattr(p, 'proj'):
                    p.proj()

        return loss

    def stepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                temp_grad = p.data.sub(state['old_weights'])
                state['old_weights'].copy_(p.data)
                p.data.add_(temp_grad)
        return loss

    def restoreStepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state['old_weights'])
        return loss


def get_closures(m, z, R, lamda, mu, enable_noise=False):
    def H(u, c):
        res = m * utils.cfft2(u * c / utils.rss(c)) - z
        e_d = (res.abs()**2).sum((1, 2, 3), keepdim=True) / 2
        return lamda * e_d + R(u)

    def nabla_c(u, c):
        rss = utils.rss(c)
        res = m * utils.cfft2(u * c / rss) - z
        inner = utils.cifft2(res)
        res_flat = utils.view_as_real_batch(inner)
        c_re_im = utils.view_as_real_batch(c)
        cross = utils.view_as_real_batch_T(
            c_re_im * (c_re_im * res_flat).sum(1, keepdim=True)
        )
        nabla_d = (inner / rss - cross / rss**3) * u
        return H(u, c), lamda * nabla_d

    def nabla_u(u, c):
        c_norm = c / utils.rss(c)
        res = m * utils.cfft2(u * c_norm) - z
        nabla_d = (utils.cifft2(res) *
                   c_norm.conj()).sum(1, keepdim=True)
        nabla_r = R.grad(u +
                         th.randn_like(u) * 7.5e-3 * enable_noise)[1]

        return (
            H(u, c),
            nabla_r + lamda * nabla_d.real,
        )

    def prox_u(u, _):
        return th.clamp_min(u, 0)

    prox_c = utils.ProxH1(*z.shape[2:], mu).cuda()

    return H, nabla_u, prox_u, nabla_c, prox_c


def apgd(
    x_init: th.Tensor,
    f_nabla: Callable[[th.Tensor], Tuple[th.Tensor, th.Tensor]],
    f: Callable[[th.Tensor], th.Tensor],
    prox: Callable[[th.Tensor, th.Tensor], th.Tensor],
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
    max_iter: int = 200,
    gamma: float = 1.,
):
    x = x_init.clone()
    x_old = x.clone()
    L = 1 * th.ones((x.shape[0], 1, 1, 1), dtype=th.float32, device=x.device)
    for i in range(max_iter):
        beta = i / (i + 3) * 1
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        n = th.randn_like(x) * 7.5e-3 * gamma
        energy, grad = f_nabla(x_bar + n)
        for _ in range(10):
            x = prox(x_bar - grad / L, 1 / L)
            dx = x - x_bar
            bound = energy + utils.inner(grad, dx, keepdim=True) \
                + L * utils.inner(dx, dx, keepdim=True) / 2
            if th.all((energy_new := f(x + n)) <= bound):
                break
            L = th.where(energy_new <= bound, L, 2 * L)
        L /= 1.5
        callback(x, i)
    return x


def ipalm(
    u_init: th.Tensor,
    c_init: th.Tensor,
    H: Callable[[th.Tensor, th.Tensor], th.Tensor],
    nabla_u: Callable[[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor]],
    prox_u: Callable[[th.Tensor, th.Tensor], th.Tensor],
    nabla_c: Callable[[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor]],
    prox_c: Callable[[th.Tensor, th.Tensor], th.Tensor],
    max_iter: int = 50,
    callback: Callable[[th.Tensor, th.Tensor, int],
                       None] = lambda u, c, i: None
) -> Tuple[th.Tensor, th.Tensor]:
    u = u_init.clone()
    c = c_init.clone()
    u_old = u.clone()
    c_old = c.clone()

    L_u = 1e0 * u.new_ones(u.shape[0], 1, 1, 1)
    L_c = 1e0 * u.new_ones(u.shape[0], 1, 1, 1)

    for it in range(max_iter):
        callback(u, c, it)
        beta = it / (it + 3)
        u_ = u + beta * (u - u_old)
        energy_u, grad_u = nabla_u(u_, c)
        u_old = u.clone()
        for _ in range(10):
            u = prox_u(u_ - grad_u / L_u, 1 / L_u)
            du = u - u_
            bound = (
                energy_u + utils.inner(grad_u, du, keepdim=True) +
                L_u / 2.0 * utils.inner(du, du, keepdim=True)
            ) * 1.01

            if th.all((energy_new := H(u, c)) <= bound):
                break
            L_u = th.where(energy_new <= bound, L_u, 2 * L_u)
        L_u /= 1.5

        c_ = c + beta * (c - c_old)
        energy_c, grad_c = nabla_c(u, c_)

        c_old = c.clone()
        for _ in range(10):
            c = prox_c(c_ - grad_c / L_c, 1 / L_c)
            dc = c - c_
            bound = (
                energy_c + utils.inner(grad_c, dc, keepdim=True) +
                L_c / 2 * utils.inner(dc, dc, keepdim=True)
            ) * 1.01

            if th.all((energy_new := H(u, c)) < bound):
                break
            L_c = th.where(energy_new < bound, L_c, 2 * L_c)
        L_c /= 1.5
        L_c.clamp_min_(.5)

    return u, c
