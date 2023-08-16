import math
import os
from pathlib import Path

import h5py
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as sl
import torch as th
from scipy import interpolate

import data
import nets
import optim
import sampling
import utils


def diffusion_comparison():
    base = Path('./evaluation') / 'diffusion-comparison'
    methods = ['map', 'mmse', 'diffusion']
    subsampling_patterns = ['random', 'cartesian', 'spiral', 'radial']

    for method in methods:
        print(method.ljust(10), end=' ')
        for pattern in subsampling_patterns:
            ref, reco = [
                th.from_numpy(
                    imageio.imread(base / pattern / f'{which}.png') / 255.
                )[None, None] for which in ['ground_truth', method]
            ]
            print(f'{utils.psnr(ref, reco).mean().item():.2f}', end=' ')
        print()


def regress_lamda(fs=False):
    num_lamdas = 10
    lamdas = th.logspace(-2, 1, num_lamdas)
    R = utils.mr_model()
    contrast = "fs" if fs else ""
    base = Path('./evaluation') / 'lambda-regression ' / f'corpd{contrast}'
    for W in [368, 372]:
        z = data.validation_data(W=W, fs=fs)
        reconstructions = th.empty((num_lamdas, z.shape[0], 1, 320, 320),
                                   device='cuda')
        m = utils.cartesian_mask(imsize=(640, W))
        f_naive = utils.mri_crop(utils.rss(utils.cifft2(m * z)))
        m = m.repeat(z.shape[0], 1, 1, 1)
        max_rss = f_naive.amax(dim=(1, 2, 3), keepdim=True)
        z /= max_rss
        z *= m

        f_init = utils.rss(utils.cifft2(z))
        c_init = utils.cifft2(z) / f_init

        initial_residual = ((m * utils.cfft2(f_init * c_init) - z).abs()**
                            2).sum((1, 2, 3), keepdim=True) / 2
        mu = 10

        for i, lam in enumerate(lamdas):
            H, nabla_u, prox_u, nabla_c, prox_c = optim.get_closures(
                m, z, R, lam, mu, True
            )
            u, _ = optim.ipalm(
                f_init,
                c_init,
                H,
                nabla_u,
                prox_u,
                nabla_c,
                prox_c,
                max_iter=100,
            )
            reconstructions[i] = utils.mri_crop(u) * max_rss

        th.save(reconstructions, base / f'reconstructions_{W}.pt')
        th.save(initial_residual, base / f'residuals_{W}.pt')
    th.save(lamdas, base / 'lamdas.pt')


def postprocess_lambda_regression(fs=False):
    path = Path(f'./evaluation/lambda-regression/corpd{"fs" if fs else ""}/')
    lamdas = th.load(path / 'lamdas.pt').squeeze()
    lambda_opt = []
    residuals = []
    for W in [368, 372]:
        gts = data.test_data_fs(W=W)[1][:12] if fs else data.test_data(W=W)[1]
        reconstructions = th.load(path / f'reconstructions_{W}.pt')
        residuals.extend(
            th.load(path / f'residuals_{W}.pt').squeeze().tolist()
        )
        for i_im, reconstruction_im in enumerate(
            reconstructions.permute(1, 0, 2, 3, 4)
        ):
            psnrs = utils.psnr(
                reconstruction_im,
                gts[i_im:i_im + 1],
                value_range=gts[i_im:i_im + 1].amax((1, 2, 3))
            )
            lambda_opt.append(lamdas[th.argmax(psnrs)])

    plt.figure()
    fit_x = np.array(residuals).reshape(-1, 1)
    reg_res = sl.LinearRegression().fit(fit_x, lambda_opt)
    print(reg_res.coef_, reg_res.intercept_)
    plt.scatter(residuals, lambda_opt)
    pred_x = np.linspace(min(residuals), max(residuals), 100)
    print(pred_x.shape)
    plt.plot(pred_x, reg_res.predict(pred_x.reshape(-1, 1)))
    plt.show()


def spline_regression(fs=False):
    R = utils.mr_model()
    gts = []
    reconstructions = []
    contrast = "fs" if fs else ""
    base = Path('./evaluation') / 'spline-fit' / f'corpd{contrast}'
    for W in [372, 368]:
        z, gt, _ = data.test_data(W=W, fs=fs)
        gts.append(gt)
        m = utils.cartesian_mask(imsize=(640, W))
        f_naive = utils.mri_crop(utils.rss(utils.cifft2(m * z)))

        m = m.repeat(z.shape[0], 1, 1, 1)
        max_rss = f_naive.amax(dim=(1, 2, 3), keepdim=True)
        z /= max_rss
        z *= m

        f_init = utils.rss(utils.cifft2(z))
        c_init = utils.cifft2(z) / f_init

        initial_residual = ((m * utils.cfft2(f_init * c_init) - z).abs()**
                            2).sum((1, 2, 3), keepdim=True) / 2
        slope, intercept, mu = (
            -35754117.0588076, 0.07383, initial_residual * 3e9
        ) if fs else (-2.71083916e+08, 0.547736379779727, 10)

        enable_noise = True
        lamda = th.maximum(
            slope * initial_residual + intercept,
            0.01 * th.ones_like(initial_residual)
        )
        H, nabla_u, prox_u, nabla_c, prox_c = optim.get_closures(
            m, z, R, lamda, mu, enable_noise
        )
        height, W = c_init.shape[2:]
        prox_c = utils.ProxH1(height, W, mu).cuda()

        u, _ = optim.ipalm(
            f_init, c_init, H, nabla_u, prox_u, nabla_c, prox_c, max_iter=100
        )
        f = utils.mri_crop(u).clamp_min(0) * max_rss
        reconstructions.append(f)

    th.save(th.cat(reconstructions).cpu(), base / 'recs.pth')
    th.save(th.cat(gts).cpu(), base / 'gts.pth')


def histogram_regression(fs=False):
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    base = Path('evaluation') / 'spline-fit' / ('corpdfs' if fs else 'corpd')
    ours, gts = [th.load(base / f'{which}.pth') for which in ['recs', 'gts']]
    x = ours.view(-1).cpu().numpy()
    indices = np.argsort(x)
    x_s = x[indices]
    y = gts.view(-1).cpu().numpy()
    y = y[indices]

    knot_numbers = 5
    x_new = np.linspace(0, 1, knot_numbers + 2)[1:-1]
    q_knots = np.quantile(x_s, x_new)
    t, c, k = interpolate.splrep(x_s, y, t=q_knots)
    # for name, val in zip(['t', 'c', 'k'], [t, c, k]):
    #     np.save(base / f'{name}.npy', val)
    yfit = interpolate.BSpline(t, c, k)(np.linspace(0, gts.max(), 100))
    ours_equalized = th.from_numpy(interpolate.BSpline(t, c, k)(x)
                                   ).view(*ours.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        np.linspace(0, gts.max(), 100), np.linspace(0, gts.max(), 100), c='r'
    )
    ax.set_xlabel('Reconstruction')
    ax.set_ylabel('Reference')
    ax.plot(np.linspace(0, gts.max(), 100), yfit, c='g')
    hb = ax.hexbin(
        ours.view(-1).cpu(), gts.view(-1).cpu(), bins='log', gridsize=200
    )
    ax.legend(['Identity', 'Spline fit'])
    ax.set_title('CORPD-FS' if fs else 'CORPD')
    axins = ax.inset_axes([0.55, 0.05, 0.42, 0.47])
    axins.hexbin(
        ours.view(-1).cpu(), gts.view(-1).cpu(), bins='log', gridsize=200
    )
    axins.plot(
        np.linspace(0, gts.max(), 100), np.linspace(0, gts.max(), 100), c='r'
    )
    axins.plot(np.linspace(0, gts.max(), 100), yfit, c='g')
    # subregion of the original image
    x1, x2, y1, y2 = -0.000005, 0.00006, -0.000005, 0.00006
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    fig.colorbar(hb, ax=ax)

    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.ticklabel_format(scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


def null_space_residual():
    ms = {
        'cartesian':
        utils.cartesian_mask(imsize=(640, 368)),
        'cartesian_4':
        utils.cartesian_mask(imsize=(640, 368), center_fraction=0.04),
    }
    path = Path(os.environ['DATASETS_ROOT']) / 'fastmri' / 'multicoil_val'
    rss = []
    fig, ax = plt.subplots(2, 2)
    # Small 'bug' in the paper: We mistakenly took the data shown in the second
    # row of Fig. 4 instead of the third, so the mask and data do not
    # line up w/ Fig. 4.
    # Radius is the number of fully sampled center lines in the kspace
    # (see the cartesian_mask function above)
    for i, (modality, filename, radius) in enumerate(
        zip(['cartesian', 'cartesian_4'], ['file1001057', 'file1002021'],
            [29, 14])
    ):
        m = ms[modality]
        file = h5py.File(path / f'{filename}.h5', 'r')
        kspace = file['kspace']
        central_slice = kspace.shape[0] // 2
        z = th.from_numpy(kspace[central_slice:central_slice + 1]).cuda()
        kspace = z.clone()

        coil_ims = utils.cifft2(kspace)
        f_naive = utils.mri_crop(utils.rss(utils.cifft2(m * z)))
        max_rss = f_naive.amax(dim=(1, 2, 3), keepdim=True)
        z /= max_rss
        z *= m

        f_init = utils.rss(utils.cifft2(z))
        c_init = utils.cifft2(z) / f_init

        initial_residual = ((m * utils.cfft2(f_init * c_init) - z).abs()**
                            2).sum((1, 2, 3), keepdim=True) / 2
        slope, intercept, mu = -2.71083916e+08, 0.547736379779727, 5
        height, W = c_init.shape[2:]

        lamda = th.maximum(
            slope * initial_residual + intercept,
            0.01 * th.ones_like(initial_residual)
        )

        R = utils.mr_model()
        H, nabla_u, prox_u, nabla_c, prox_c = optim.get_closures(
            m, z, R, lamda, mu, True
        )

        u, c = optim.ipalm(
            f_init,
            c_init,
            H,
            nabla_u,
            prox_u,
            nabla_c,
            prox_c,
            max_iter=100,
        )

        def normalize(c):
            return c / th.sqrt((c.conj() * c).sum(1, keepdims=True))

        def proj_ortho_compl(ims, cs):
            return cs * (cs.conj() * ims).sum(1, keepdims=True) - ims

        def _rss(x):
            return np.rot90(utils.rss(x).cpu().numpy().squeeze(), k=2)

        ours_proj = _rss(proj_ortho_compl(coil_ims, normalize(c)))
        espirit_proj = _rss(
            proj_ortho_compl(coil_ims, normalize(utils.espirit(z, radius)))
        )

        # Maximum over all estimations (4% and 8%, ours and espirit) for
        # consistent plotting
        max_all = 2.9563955e-05
        ax[0, i].imshow(ours_proj / max_all, cmap='gray', vmin=0, vmax=1)
        ax[1, i].imshow(espirit_proj / max_all, cmap='gray', vmin=0, vmax=1)

    plt.show()


def parallel_imaging(modality, fs=False):
    Ssim = utils.SSIM().cuda().float()
    m_init = {
        'cartesian': utils.cartesian_mask(),
        'cartesian_4': utils.cartesian_mask(center_fraction=0.04),
        'cartesian_rot90': utils.cartesian_mask(vertical=False),
        'radial': utils.radial_mask((640, 372), 45, np.pi / 45),
        'gaussian2d': utils.gaussian2d_mask((640, 372), 8),
    }[modality]

    methods = ['zf', 'vn', 'tv', 'ours', 'ours_mmse']
    psnrs, ssims, nmses = [{
        method: []
        for method in methods
    } for _ in range(3)]
    step = 10

    which = 'fs' if fs else ''
    spline_path = Path(f'./evaluation/spline-fit/corpd{which}/')
    t, c, k = [np.load(spline_path / f'{n}.npy') for n in ['t', 'c', 'k']]
    equalizer = interpolate.BSpline(t, c, k)
    varnet = utils.varnet_model()

    for W in [372, 368]:
        if W == 368:
            m_init = m_init[..., 2:-2]

        m_vn = m_init.clone()
        zs, gts, _ = data.test_data(W=W, fs=fs)
        for start in range(0, len(zs), step):
            z = zs[start:start + step]
            gt = gts[start:start + step]

            # VN requires additional dimension on mask b/c it computes on the
            # "real view" (real/imag in last dim)
            f_vn = utils.mri_crop(
                varnet(
                    th.view_as_real(z * m_vn),
                    m_vn[..., None],
                )[:, None]
            )

            m = m_init.repeat(z.shape[0], 1, 1, 1)
            f_naive = utils.mri_crop(utils.rss(utils.cifft2(m_init * z)))
            max_rss = f_naive.amax(dim=(1, 2, 3), keepdim=True)
            z /= max_rss
            z *= m

            f_init = utils.rss(utils.cifft2(z))
            c_init = utils.cifft2(z) / f_init

            initial_residual = ((m * utils.cfft2(f_init * c_init) - z).abs()**
                                2).sum((1, 2, 3), keepdim=True) / 2
            slope, intercept, mu = ((
                -35754117.0588076, 0.07383, initial_residual * 3e9
            ) if fs else (-2.71083916e+08, 0.547736379779727, 10))
            height, W = c_init.shape[2:]

            # Clamping needed otherwise it may go negative
            lamda = th.maximum(
                slope * initial_residual + intercept,
                0.01 * th.ones_like(initial_residual)
            )

            # We inject when evauation the gradient of the regularizer
            # This noise does not leak into the iterates
            enable_noise = True
            R = utils.mr_model()
            H, nabla_u, prox_u, nabla_c, prox_c = utils.get_closures(
                m, z, R, lamda, mu, True
            )

            u, c = optim.ipalm(
                f_init,
                c_init,
                H,
                nabla_u,
                prox_u,
                nabla_c,
                prox_c,
                max_iter=100
            )

            f_star = th.from_numpy(
                equalizer((utils.mri_crop(u) *
                           max_rss).clamp_min(0).view(-1).cpu().numpy())
            ).cuda().view(*gt.shape).float()

            H, nabla_u, prox_u, nabla_c, prox_c = utils.get_closures(
                m, z, R, lamda, mu
            )
            c_tilde = c / utils.rss(c)

            def accumulate(u, _):
                nonlocal num, mean
                num = num + 1
                delta = u - mean
                mean += delta / num

            num = 0
            mean = th.zeros_like(f_init)

            burn = sampling.ula(u, lambda x: nabla_u(x, c_tilde)[1], n=2_0)
            _ = sampling.ula(
                burn,
                lambda x: nabla_u(x, c_tilde)[1],
                n=5_0,
                callback=accumulate
            )
            f_mmse = th.from_numpy(
                equalizer((utils.mri_crop(mean) *
                           max_rss).clamp_min(0).view(-1).cpu().numpy())
            ).cuda().view(*gt.shape).float()

            # Found by hand
            mult = 1 / 0.003
            lamda = (slope * initial_residual + intercept) * mult
            mu = 100
            H, nabla_u, prox_u, nabla_c, prox_c = utils.get_closures(
                m, z, nets.CharbTV(eps=1e-3), lamda, mu
            )

            u, c = optim.ipalm(
                f_init,
                c_init,
                H,
                nabla_u,
                prox_u,
                nabla_c,
                prox_c,
                max_iter=100
            )
            f_tv = th.from_numpy(
                equalizer((utils.mri_crop(u) *
                           max_rss).clamp_min(0).view(-1).cpu().numpy())
            ).cuda().view(*gt.shape).float()

            for im, name in zip([f_naive, f_tv, f_vn, f_star, f_mmse],
                                methods):
                ranges = gt.amax((1, 2, 3), keepdim=True)
                psnrs[name].extend(
                    utils.psnr(im, gt, ranges).squeeze().tolist()
                )
                ssims[name].extend(Ssim(im, gt, ranges).squeeze().tolist())
                nmses[name].extend(
                    (100 * utils.nmse(gt, im)).squeeze().tolist()
                )

    print(modality)
    fstr = '& \\gls{{psnr}} (\\si{{\\decibel}}) & {} & {} & {} & {} & {}\\\\\n'
    fstr += '& & & \\gls{{nmse}} & {} & {} & {} & {} & {}\\\\\n'
    fstr += '& & & \\gls{{ssim}} & {} & {} & {} & {} & {}\\\\'
    print(
        fstr.format(
            *[
                np.mean(metric[method]) for metric in [psnrs, nmses, ssims]
                for method in methods
            ]
        )
    )


def grid_search_synthetic():
    for sampling_pattern, params in zip(
        ['cartesian', 'spiral', 'radial', 'random'],
        [(320, 0.1, 4), (2, 8000, 40), (240, 25), (320, 3)],
    ):
        _, A, f, _, _ = data.MRCentralSliceVal(
            mask=sampling_pattern,
            mask_params=params,
            num=10,
        ).data()
        p = A @ f

        def callback(lamda):
            f_init = th.zeros_like(A.H @ (A @ f))
            h, nabla_h = utils.data_l2(lamda, A, p)
            reco = utils.tgv(
                th.zeros_like(f_init), h, nabla_h, it=1000, beta=1 / lamda
            )
            psnr = utils.psnr(utils.rss(reco), utils.rss(f))
            print(psnr, lamda)
            return psnr

    best, argbest = utils.grid_search(np.linspace(5, 7e3, 20), callback)
    print(best, argbest)


lamdas = {
    'ours': {
        'cartesian': 2.15,
        'random': 0.22,
        'radial': 4.64,
        'spiral': 2.15,
    },
    'tv': {
        'cartesian': 784.76,
        'random': 183.30,
        'radial': 3359.82,
        'spiral': 2335.72,
    },
}


def single_coil_synthetic():
    Unet = utils.unet_model()
    Ssim = utils.SSIM().cuda()
    y = data.synthetic_data()
    y = y / y.amax(dim=(1, 2, 3), keepdim=True)

    for sampling_pattern, mask_params in zip(
        ['random', 'cartesian', 'spiral', 'radial'],
        [((320, 320), 3), ((320, 320), ), (1.5, 40_000, 80), ((320, 320), 45)],
    ):
        m = th.fft.fftshift({
            'cartesian': utils.cartesian_mask,
            'radial': utils.radial_mask,
            'spiral': utils.spiral_mask,
            'random': utils.random_mask,
        }[sampling_pattern](*mask_params).cuda().to(th.float32))

        A = utils.AsMatrix(
            lambda x: m * th.fft.fft2((x + 0j), norm='ortho'),
            lambda y: th.fft.ifft2(m * y, norm='ortho').real,
        )

        p = A @ y
        zero_filled = A.H @ p
        zero_filled = zero_filled / zero_filled.amax((1, 2, 3), keepdim=True)
        f_init = y.view(*y.shape[:2], y.shape[2] *
                        y.shape[3])[...,
                                    th.randperm(y.shape[2] * y.shape[3])].view(
                                        y.shape
                                    ).clone()

        def f_nabla(f):
            dterm = lamda * ((A @ f - p).abs()**2).sum((1, 2, 3)) / 2
            nabla_dterm = lamda * (A.H @ (A @ f - p))
            reg, nabla_reg = R.grad(f)
            return (
                dterm[:, None, None, None] + reg,
                nabla_dterm + nabla_reg,
            )

        R = utils.mr_model()
        lamda = lamdas['ours'][sampling_pattern]
        ours = optim.apgd(
            f_init,
            f_nabla,
            lambda x: f_nabla(x)[0],
            lambda x, _: x,
            max_iter=200,
        )

        def nabla(f):
            return lamda * (A.H @ (A @ f - p)).real + R.grad(f)[1]

        # (each * samples + burn_in) Langevin iterations in total
        samples = 10
        each = 15
        burn_in = 1_000

        def accumulate(f, i):
            nonlocal n, mmse, m2
            idx = (i - burn_in) // each
            if idx < samples and i > burn_in and i % each == 0:
                n = n + 1
                delta = f - mmse
                mmse += delta / n
                m2 += delta * (f - mmse)

        n = 0
        # MMSE and variance
        mmse = th.zeros_like(y)
        m2 = th.zeros_like(y)

        # Initialize with MAP estimate
        _ = sampling.ula(
            ours,
            nabla=nabla,
            callback=accumulate,
            n=burn_in + each * samples,
        )

        unet = utils.call_unet(
            Unet,
            th.fft.ifft2(p, norm='ortho').abs(),
        )
        # Changes 'R' and 'lamda' in the closure
        R = nets.CharbTV(eps=1e-3)
        lamda = lamdas['tv'][sampling_pattern]
        tv = optim.apgd(
            f_init,
            f_nabla,
            lambda x: f_nabla(x)[0],
            lambda x, _: x,
            max_iter=200,
            gamma=0,
        )
        # This only improves TV
        tv = tv / tv.amax((1, 2, 3), keepdim=True)

        fstr = '\\multirow{{3}}{{*}}{{{}}}'
        fstr += '& \\gls{{psnr}} & {} & {} & {} & {} & {} \\\\\n'
        fstr += '& \\gls{{nmse}} & {} & {} & {} & {} & {} \\\\\n'
        fstr += '& \\gls{{ssim}} & {} & {} & {} & {} & {} \\\\\n'
        print(
            fstr.format(
                *[sampling_pattern] + [
                    metric(y, method).mean().item()
                    for metric in [utils.psnr, utils.nmse, Ssim]
                    for method in [zero_filled, tv, unet, ours, mmse]
                ],
            )
        )
        print('\\midrule')


def generate_trajectories():
    R = utils.mr_model()
    n_samples = 10
    y = data.synthetic_data()[:n_samples]
    # Initial point: Scrambling along pixel dimension
    f = y.view(*y.shape[:2],
               y.shape[2] * y.shape[3])[...,
                                        th.randperm(y.shape[2] *
                                                    y.shape[3])].view(y.shape
                                                                      ).clone()
    N = 10000
    each = 10
    ims = th.empty((N // each, 10, 320, 320)).cuda()

    def callback(x, i):
        if i % each == 0:
            ims[i // each] = x[:, 0]

    _ = sampling.ula(f, lambda x: R.grad(x)[1], n=N, callback=callback)
    th.save(ims, Path('./out') / 'mri' / 'trajectories' / 'trajectories.pth')


def visualize_trajectories():
    R = utils.mr_model()
    path = Path('./out') / 'mri' / 'trajectories'
    trajs = th.load(path / 'trajectories.pth',
                    map_location='cpu').permute(1, 0, 2, 3)
    steps = 100
    T = 7
    index_list = [0, 20, 50, 200, 500, 999]
    for i_traj, traj in enumerate(trajs):
        for index in index_list:
            imageio.imwrite(
                path / f'{i_traj}' / f'{index}.png',
                (np.clip(trajs[i_traj, index].cpu().numpy(), 0, 1) *
                 255).astype(np.uint8)
            )

    for i_traj, traj in enumerate(trajs):
        traj = traj.cuda()
        M = traj[:-1] - traj[-1]
        mean = M.mean(0, keepdim=True)
        M -= mean
        eigvecs = th.linalg.svd(M.view(M.shape[0], -1),
                                full_matrices=False)[-1][:2]
        eigvecs = eigvecs.view(2, 1, 320, 320)
        minmax1 = th.aminmax((M * eigvecs[0]).sum((1, 2)))
        minmax2 = th.aminmax((M * eigvecs[1]).sum((1, 2)))
        alpha = th.linspace(minmax1[0] - 20, minmax1[1] + 20, steps).cuda()
        beta = th.linspace(minmax2[0] - 20, minmax2[1] + 20, steps).cuda()
        a, b = th.meshgrid(alpha, beta, indexing='xy')
        a_flat = a.reshape(-1, 1, 1)
        b_flat = b.reshape(-1, 1, 1)
        energies = th.zeros((steps**2, ))
        chunk = 100
        for i in range(math.ceil(steps**2 / chunk)):
            start = chunk * i
            end = min(chunk * (i + 1), len(energies))
            current = traj[-1][None] + mean.view(1, 320, 320) + \
                a_flat[start:end] * eigvecs[0] + b_flat[start:end] * eigvecs[1]
            energies[start:end] = R.energy(current[:, None]).squeeze() / T

        energies_traj = th.zeros(len(traj))
        xs = ((traj - traj[-1:] - mean.view(320, 320)[None]) * eigvecs[0]).sum(
            (1, 2)
        )
        ys = ((traj - traj[-1:] - mean.view(320, 320)[None]) * eigvecs[1]).sum(
            (1, 2)
        )
        for i in range(math.ceil(len(traj) / chunk)):
            start = chunk * i
            end = min(chunk * (i + 1), len(energies))
            current = traj[-1][None] + mean.view(1, 320, 320) \
                + xs.view(-1, 1, 1)[start:end] * eigvecs[0] \
                + ys.view(-1, 1, 1)[start:end] * eigvecs[1]
            energies_traj[start:end] = R.energy(current[:, None]).squeeze() / T

        plt.rcParams.update({"text.usetex": True, "font.size": 12})
        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=160,
            subplot_kw={
                "projection": "3d",
                "computed_zorder": False,
            }
        )
        ax.set_proj_type('ortho')
        Z = th.exp(-energies).view(steps, steps).cpu()
        norm = plt.Normalize(Z.min(), Z.max())
        colors = plt.colormaps['coolwarm'](norm(Z))
        ax.plot_surface(
            a.cpu(),
            b.cpu(),
            th.exp(-energies).view(steps, steps).cpu(),
            facecolors=colors,
            rstride=2,
            cstride=2,
        )
        ax.view_init(elev=43, azim=-48)
        ax.plot(
            xs.cpu().numpy(),
            ys.cpu().numpy(),
            th.exp(-energies_traj).cpu().numpy(),
            zorder=1.,
            linewidth=2,
            c='xkcd:gold'
        )
        ax.scatter(
            xs.cpu()[index_list],
            ys.cpu()[index_list],
            th.exp(-energies_traj)[index_list],
            c='xkcd:gold',
            zorder=2,
            s=50
        )
        ax.set_xlabel('$\\xi_1$')
        ax.set_ylabel('$\\xi_2$')
        plt.savefig(path / f'{i_traj}' / 'pdf.pdf')


subsampling_patterns = [
    'cartesian', 'cartesian_rot90', 'cartesian_4', 'radial', 'gaussian2d'
]

if __name__ == '__main__':
    th.manual_seed(0)
    with th.no_grad():
        # visualize_trajectories()  # Fig. 2
        single_coil_synthetic()  # Tab. 1, Fig. 3
        # diffusion_comparison()  # Tab. 2, Fig. 9
        # for pattern in subsampling_patterns:  # Tab. 3, Fig. 4 and 5
        #     parallel_imaging(pattern, True)
        # null_space_residual()  # Fig. 7
        # histogram_regression(True)  # Fig. 8
        # regress_lamda(which='fs')
        # postprocess_lambda_regression()
        # spline_regression(False)
        # synthetic_posterior_sampling()
        pass
