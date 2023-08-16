import torch as th
import numpy as np


__all__ = ['Conv2d', 'ConvScale2d', 'ConvScaleTranspose2d', 'Upsample2x2']


class Conv2d(th.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, invariant=False,
        stride=1, dilation=1, groups=1, bias=False,
        zero_mean=False, bound_norm=False, positive=False, pad=True
    ):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = th.nn.Parameter(
            th.zeros(out_channels)
        ) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.positive = positive
        self.padding = 0
        self.pad = pad

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            self.weight = th.nn.Parameter(
                th.empty(out_channels, in_channels, 1,  3)
            )
            self.register_buffer('mask', th.from_numpy(
                np.asarray([1, 4, 4], dtype=np.float32)[None, None, None, :]
            ))
        else:
            self.weight = th.nn.Parameter(th.empty(
                out_channels, in_channels, self.kernel_size,  self.kernel_size
            ))
            self.register_buffer('mask', th.from_numpy(np.ones(
                (self.kernel_size, self.kernel_size), dtype=np.float32
            )[None, None, :, :]))
        # insert them using a normal distribution
        th.nn.init.normal_(
            self.weight.data, 0.0,
            np.sqrt(1 / np.prod(in_channels * kernel_size ** 2))
        )

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3)
            # define a projection

            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = th.sum(
                        self.weight.data * self.mask,
                        self.weight.reduction_dim,
                        True
                    ) / (self.in_channels*self.kernel_size**2)
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = th.sum(
                        self.weight.data**2 * self.mask,
                        self.weight.reduction_dim,
                        True
                    ).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            th.max(norm, th.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)

        elif positive:
            self.weight.reduction_dim = (1, 2, 3)
            self.weight.proj = lambda: self.weight.data.clamp_(min=1e-2)
            self.weight.data.fill_(1e-1)

    def forward(self, x):
        pad = self.weight.shape[-1] // 2
        if self.pad and pad > 0:
            x = th.nn.functional.pad(x, (pad, pad, pad, pad))
        return th.nn.functional.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

    def backward(self, x, output_padding=0):
        x = th.nn.functional.conv_transpose2d(
            x, self.weight, self.bias, self.stride,
            self.padding, output_padding, self.groups, self.dilation
        )
        pad = self.weight.shape[-1] // 2
        if self.pad and pad > 0:
            x = x[..., pad:-pad, pad:-pad]
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is not None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        if self.positive:
            s += ", positive={positive}"
        return s.format(**self.__dict__)


class ConvScale2d(Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, invariant=False,
        groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False
    ):
        super(ConvScale2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, invariant=invariant, stride=stride,
            dilation=1, groups=groups, bias=bias,
            zero_mean=zero_mean, bound_norm=bound_norm
        )

    def get_weight(self):
        weight = super().get_weight()
        return weight


class ConvScaleTranspose2d(ConvScale2d):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, invariant=False,
        groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False
    ):
        super(ConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, invariant=invariant, groups=groups,
            stride=stride, bias=bias,
            zero_mean=zero_mean, bound_norm=bound_norm
        )

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)


class Upsample2x2(Conv2d):
    def __init__(
        self, in_channels, out_channels, zero_mean=False, bound_norm=False
    ):
        super(Upsample2x2, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            invariant=False, groups=1, stride=1, bias=None,
            zero_mean=zero_mean, bound_norm=bound_norm
        )

        self.in_channels = in_channels

        # define the interpolation kernel for 2x2 upsampling
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k = np_k @ np_k.T
        np_k /= np_k.sum()
        np_k *= 4  # TODO: check!
        np_k = np.reshape(np_k, (1, 1, 5, 5))
        np_k = np.tile(np_k, (self.in_channels, 1, 1, 1))
        self.register_buffer('kernel', th.from_numpy(np_k))

    def forward(self, x, output_shape=None):
        pad = self.kernel.shape[-1]//2//2

        # determine the amount of padding
        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*2+1),
                output_shape[3] - ((x.shape[3]-1)*2+1)
            )
        else:
            output_padding = 0

        # x = optoth.pad2d.pad2d(x, (pad, pad, pad, pad), mode='symmetric')
        # apply the upsampling kernel
        x = th.nn.functional.conv_transpose2d(
            x, self.kernel, stride=2, padding=4 * pad,
            output_padding=output_padding, groups=self.in_channels
        )
        # apply the 1x1 convolution
        return x  # super().forward(x)

    def backward(self, x):
        # 1x1 convolution backward
        # x = super().backward(x)
        # upsampling backward
        pad = self.kernel.shape[-1] // 2 // 2
        x = th.nn.functional.conv2d(
            x, self.kernel, stride=2, padding=4 * pad, groups=self.in_channels
        )
        # return optoth.pad2d.pad2d_transpose(
        #     x, (pad, pad, pad, pad), mode='symmetric'
        # )
