# -*- coding: UTF-8 -*-
"""Script for calculating connectivity measures."""

import os
import math
import h5py
import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from mne.connectivity import phase_slope_index


def _try_epoching(dataset=None, fs=500, transpose=True):
    if dataset is None:
        test_timepoints = 1100
        dataset = np.ndarray(shape=(test_timepoints, 2))
    nrows = dataset.shape[0]
    ncols = dataset.shape[1]
    new_constant = int(fs * 1)  # min requirement infraslow PSI is 500s
    n_splits = math.ceil(nrows/new_constant)
    indexer = KFold(n_splits=n_splits)
    if transpose is False:
        epoched_data = np.ndarray(shape=(n_splits, new_constant, ncols))
    else:
        epoched_data = np.ndarray(shape=(n_splits, ncols, new_constant))
    for i, (_, ts) in enumerate(indexer.split(dataset)):
        print(ts)
        segment = dataset[ts, :]
        if segment.shape[0] < new_constant:
            epoch = np.pad(
                segment,
                pad_width=((0, int(fs-segment.shape[0])), (0, 0)),
                constant_values=0)
        else:
            epoch = segment
        if transpose is True:
            epoch = epoch.T
        epoched_data[i, :, :] = epoch
        del epoch
    return epoched_data


def epoch_MEG(rois=None, fs=500, transpose=True):
    data_dir = utils.ProjectData.data_dir
    meg_subj, meg_sess = utils.ProjectData.meg_metadata
    epoch_file = os.path.join(data_dir, 'MEG_epoched.hdf5')
    out = h5py.File(epoch_file, 'a')
    for sess in meg_sess:
        for subj in meg_subj:
            path = subj + '/' + sess
            if path in out:
                print(path)
                continue

            key = subj + '/MEG/' + sess + '/timeseries'
            print('Epoching %s' % str(key))
            dataset = utils.index_hcp_raw(key=key, rois=rois)
            nrows = dataset.shape[0]
            ncols = dataset.shape[1]
            n_splits = math.ceil(nrows/fs)
            indexer = KFold(n_splits=n_splits)
            d = np.float32
            if transpose is False:
                epoched_data = np.ndarray(shape=(n_splits, fs, ncols), dtype=d)
            else:
                epoched_data = np.ndarray(shape=(n_splits, ncols, fs), dtype=d)
            for i, (_, ts) in enumerate(indexer.split(dataset)):
                segment = dataset[ts, :]
                if segment.shape[0] < fs:
                    epoch = np.pad(
                        segment,
                        pad_width=((0, int(fs-segment.shape[0])), (0, 0)),
                        constant_values=0)
                else:
                    epoch = segment
                if transpose is True:
                    epoch = epoch.T
                epoched_data[i, :, :] = epoch
                del epoch, segment
            del dataset
            grp = out.require_group(path)
            grp.create_dataset('epochs', data=epoched_data, compression='lzf')

            del epoched_data
    out.close()


def effective_connectivity(start_rois):
    data_dir = utils.ProjectData.data_dir
    subjects, sessions = utils.ProjectData.meg_metadata
    glasser_rois = utils.ProjectData.glasser_rois

    start_indices, end_indices, connections = [], [], []
    if type(start_rois) == str:
        start_rois = [start_rois]
    for sr in start_rois:
        for g, gr in enumerate(glasser_rois):
            if sr == gr:
                idxs = [g] * len(glasser_rois)
                start_indices.extend(idxs)
                end_indices.extend(np.arange(len(glasser_rois)))
                connections.append('%s %s' % (sr, gr))
    start_indices = np.array(start_indices)
    indices = (start_indices, end_indices)
    # epoch_file = os.path.join(data_dir, 'MEG_epoched.hdf5')
    for sess in sessions:
        for subj in subjects:
            key = subj + '/MEG/' + sess + '/timeseries'
            print('Epoching %s' % str(key))
            dataset = utils.index_hcp_raw(key=key)
            epoch_data = _try_epoching(dataset)
            del dataset
            # f = h5py.File(epoch_file, 'r')
            # epoch_data = f[subj][sess]['epochs'][...]
            # f.close()
            eff_con, _, _, _, _ = phase_slope_index(
                data=epoch_data,
                indices=indices,
                fmin=8,
                fmax=12,
                mode='fourier',
                sfreq=500,
                verbose=0,
            )
            del epoch_data
            print(eff_con)
            break
        break


if __name__ == "__main__":
    # epoch_MEG()
    effective_connectivity(start_rois=['p24pr_L', 'p24pr_R'])
