import os
from pathlib import Path
from typing import List, Tuple

import h5py
import pandas as pd
import torch as th


def training_data():
    path = os.path.join(
        os.environ['DATASETS_ROOT'], 'fastmri', 'multicoil_train'
    )
    filenames = open('./split/corpd/train.txt', 'r').read().splitlines()
    images = th.empty((len(filenames) * 11, 1, 320, 320))

    for i_f, filename in enumerate(filenames):
        file = h5py.File(os.path.join(path, filename), 'r')
        reconstructions = file['reconstruction_rss']
        central_slice = reconstructions.shape[0] // 2
        for i_slice in range(-5, 6):
            rec_slice = th.from_numpy(
                reconstructions[central_slice + i_slice]
            )[None]
            images[11 * i_f + i_slice + 5] = rec_slice / rec_slice.max()

    return images.cuda()


def validation_data(W: int = 368, fs=False):
    # We did not split the fs data in validation and test
    if fs:
        return _val_or_test_data('test', W, fs)
    return _val_or_test_data('validation', W)


def test_data(W: int = 368, fs: bool = False):
    return _val_or_test_data('test', W, fs)


# Stack data for with 368 and 372 for synthetic experiments,
# where only the 320x320 ground truth is needed
def synthetic_data():
    return th.cat([test_data(W)[1] for W in [368, 372]])


def _val_or_test_data(split, W: int = 368, fs: bool = False):
    which = 'corpdfs' if fs else 'corpd'
    path = Path(os.environ['DATASETS_ROOT']) / 'fastmri' / 'multicoil_val'
    test_samples = open(f'./split/{which}/{split}.txt',
                        'r').read().splitlines()
    sz = 320
    num = len(test_samples)
    # We dont know how many samples we will get, so we have to allocate
    kspaces = th.empty((num, 15, 640, W), dtype=th.complex64)
    ground_truth = th.empty((num, 1, sz, sz))
    # We will keep track of how many samples we have added
    added = 0
    filenames = []

    for fname in test_samples:
        file = h5py.File(path / fname, 'r')
        kspace = file['kspace']
        if kspace.shape[-1] != W:
            continue
        central_slice = kspace.shape[0] // 2
        for i_slice in range(1):
            kspaces[added] = th.from_numpy(kspace[central_slice + i_slice])
            ground_truth[added] = th.from_numpy(
                file['reconstruction_rss'][central_slice + i_slice]
            )
            filenames.append(fname)
            added += 1

    return kspaces[:added].cuda(), ground_truth[:added].cuda(), filenames


class CherryDatasetCoils():
    def __init__(
        self,
        num: int = 25,
        W: int = 368,
    ):
        path = Path(os.environ['DATASETS_ROOT']) / 'fastmri' / 'multicoil_val'
        # test_samples = [
        #     "file1001057.h5", "file1002021.h5",
        #     "file1001598.h5", "file1001983.h5",
        #     "file1002257.h5"
        # ]
        test_samples = [
            "file1002021.h5", "file1001598.h5", "file1001983.h5",
            "file1002257.h5"
        ]
        sz = 320
        num = min(len(test_samples), num)
        kspaces = th.empty((num, 15, 640, W), dtype=th.complex64)
        ground_truth = th.empty((num, 1, sz, sz))
        added = 0
        filenames = []

        for fname in test_samples:
            if added == num:
                break
            file = h5py.File(path / fname, 'r')
            kspace = file['kspace']
            print(kspace.shape[-1])
            if kspace.shape[-1] != W:
                continue
            central_slice = kspace.shape[0] // 2
            kspaces[added] = kspace[central_slice]
            ground_truth[added] = file['reconstruction_rss'][central_slice]
            filenames.append(fname)
            added += 1

        self.kspace = th.from_numpy(kspaces[:added])
        self.ground_truth = th.from_numpy(ground_truth[:added])
        self.fnames = filenames

    def data(self) -> Tuple[th.Tensor, th.Tensor, List[str]]:
        return self.kspace.cuda(), self.ground_truth.cuda(), self.fnames
