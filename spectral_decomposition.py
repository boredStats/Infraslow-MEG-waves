# -*- coding: UTF-8 -*-
"""Script for spectral decomposition of MEG data, and associated analyses.

Note: these functions require timeseries extracted using the Glasser 2016
    atlas. See Methods section for more.
"""
import os
import h5py
import utils
import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr
from scipy.signal import butter, hilbert, sosfilt
from astropy.stats.circstats import circcorrcoef as circ_corr


def _index_hcp_raw(key, indices=None):
    data_dir = utils.ProjectData.data_dir
    hcp_file = os.path.join(data_dir, 'multimodal_HCP.hdf5')

    database = h5py.File(hcp_file, 'r+')
    if indices:
        dset = database[key][:, indices]
    else:
        dset = database[key][...]
    database.close

    return dset


def _calc_psd(timeseries, bandpass, fs=500):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_amp = np.fft.rfft(timeseries, axis=0)
    fft_power = np.absolute(fft_amp) ** 2  # Squared for psd
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(timeseries), 1.0 / fs)
    freq_ix = np.where((fft_freq >= bandpass[0]) &
                       (fft_freq <= bandpass[1]))[0]

    avg_power = np.mean(fft_power[freq_ix])
    return avg_power


def _get_MEG_psd(band='BOLD bandpass', output_file=None):
    subjects, sessions = utils.ProjectData.meg_metadata
    rois = utils.ProjectData.glasser_rois
    band_dict = utils.ProjectData.freq_bands
    bandpass = band_dict[band]
    fs = 500

    session_data = {}
    for session in sessions:
        psd_df = pd.DataFrame(index=subjects, columns=rois)
        for subject in subjects:
            prog = "%s - %s" % (session, subject)
            print('%s: Calculating PSD for %s' % (utils.ctime(), prog))
            key = subject + '/MEG/' + session + '/timeseries'
            for r, roi in enumerate(rois):
                timeseries = _index_hcp_raw(key, r)
                psd_df.loc[subject][roi] = _calc_psd(timeseries, bandpass)
        session_data[session] = session_df

    if output_file:
        save_xls(session_data, output_file)
    return session_data


def _spectra_decomp(data, fs, phase_band=None, amp_band=None):
    def _butter_filter(timeseries, fs, cutoffs, btype='band', order=4):
        nyquist = fs/2
        butter_cut = np.divide(cutoffs, nyquist)
        sos = butter(order, butter_cut, output='sos', btype=btype)
        return sosfilt(sos, timeseries)

    phase_banded = _butter_filter(data, fs, phase_band)
    phase_hilbert = hilbert(phase_banded)
    phase_data = np.angle(phase_hilbert)

    amp_banded = butter_filter(data, fs, amp_band)
    amp_hilbert = hilbert(amp_banded)
    amp_data = np.absolute(amp_hilbert)

    return phase_data, amp_data


def _get_phase_amp(ts_data, bandpass):
    rois = ProjectData.glasser_rois
    fs = 500
    ts_len = len(ts_data[rois[0]])
    phase_mat = np.ndarray(shape=[ts_len, len(rois)])
    amp_mat = np.ndarray(shape=[ts_len, len(rois)])

    for r, roi in enumerate(rois):
        phase, amp = _spectra_decomp(ts_data[:, r], fs, bandpass, bandpass)
        phase_mat[:, r] = phase
        amp_mat[:, r] = amp

    return phase_mat, amp_mat


def _calc_phase_amp():
    ProjectData = utils.ProjectData
    data_dir = ProjectData.data_dir

    rois = ProjectData.glasser_rois
    bands = ProjectData.freq_bands
    meg_subj, meg_sess = ProjectData.meg_metadata
    outfile = os.path.join(data_dir, 'MEG_phase_amp_data.hdf5')

    for sess in meg_sess:
        for subj in meg_subj:
            for band in bands:
                bandpass = bands[band]
                out_file = h5py.File(data_path)
                group_path = subj + '/' + sess + '/' + band
                if group_path in out_file:
                    continue

                key = subj + '/MEG/' + sess + '/timeseries'
                dset = _index_hcp_raw(key)
                phase_mat, amp_mat = _get_phase_amp(dset, bandpass)

                grp = out_file.require_group(group_path)
                grp.create_dataset(
                    'phase_data',
                    data=phase_mat,
                    compression='lzf')
                grp.create_dataset(
                    'amplitude_data',
                    data=amp_mat,
                    compression='lzf')
                out_file.close()


def _circ_line_corr(ang, line):
    # Correlate periodic data with linear data
    n = len(ang)
    rxs = pearsonr(line, np.sin(ang))
    rxs = rxs[0]
    rxc = pearsonr(line, np.cos(ang))
    rxc = rxc[0]
    rcs = pearsonr(np.sin(ang), np.cos(ang))
    rcs = rcs[0]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))
    pval = 1 - chi2.cdf(n*(rho**2), 1)
    # standard_error = np.sqrt((1-r_2)/(n-2))

    return rho, pval  # ,standard_error


def _calc_pac():
    ProjectData = utils.ProjectData
    data_dir = ProjectData.data_dir

    rois = ProjectData.glasser_rois
    bands = ProjectData.freq_bands
    meg_subj, meg_sess = ProjectData.meg_metadata

    data_path = os.path.join(data_dir, 'MEG_phase_amp_data.hdf5')
    coupling_path = os.path.join(data_dir, 'MEG_phase_amp_coupling.hdf5')

    phase_bands = list(bands)
    amp_bands = list(bands)

    for sess in meg_sess:
        for subj in meg_subj:
            data_file = h5py.File(data_path, 'r')
            subj_data = data_file.get(subj + '/' + sess)
            for r, roi in enumerate(rois):
                cfc_file = h5py.File(coupling_path)
                group_path = sess + '/' + subj + '/' + roi
                if group_path in cfc_file:
                    continue  # check if work has already been done

                r_mat = np.ndarray(shape=(len(phase_bands), len(amp_bands)))
                p_mat = np.ndarray(shape=(len(phase_bands), len(amp_bands)))
                for phase_index, phase_band in enumerate(phase_bands):
                    p_grp = subj_data.get(phase_band)
                    phase = p_grp.get('phase_data')[:, r]
                    for amp_index, amp_band in enumerate(amp_bands):
                        a_grp = subj_data.get(amp_band)
                        amp = a_grp.get('amplitude_data')[:, r]

                        r_val, p_val = _circ_line_corr(phase, amp)
                        r_mat[phase_index, amp_index] = r_val
                        p_mat[phase_index, amp_index] = p_val

                out_group = cfc_file.require_group(group_path)
                out_group.create_dataset(
                    'r_vals',
                    data=r_mat,
                    compression=comp)
                out_group.create_dataset(
                    'p_vals',
                    data=p_mat,
                    compression=comp)
                cfc_file.close()

            data_file.close()


def sort_roi_names(rois):
    """Sort roi names based on numpy matrix indexing.

    Use this when using subsets of ROIs in analyses and indexing HCP datasets.
    """
    glasser_rois = utils.ProjectData.glasser_rois
    roi_indices = []
    for roi in rois:
        true_index = glasser_rois.index(roi)
        roi_indices.append(true_index)
    sorted_indices = sorted(roi_indices)
    sorted_roi_names = []
    for si in sorted_indices:
        ordered_name = glasser_rois[si]
        sorted_roi_names.append(ordered_name)

    return sorted_indices, sorted_roi_names


def _calc_ppc(band, output_file=None, rois=None):
    ProjectData = utils.ProjectData
    data_dir = ProjectData.data_dir

    subjects, sessions = ProjectData.meg_metadata

    phase_amp_data = os.path.join(data_dir, 'MEG_phase_amp_data.hdf5')
    if output_file is None:
        output_file = os.path.join(data_dir, 'MEG_phase_phase_coupling.hdf5')
    if rois is None:
        rois = ProjectData.glasser_rois  # Not recommended, specify ROIs

    roi_indices, sorted_roi_names = sort_roi_names(rois)

    for sess in sessions:
        for subj in subjects:
            print('%s: Running %s %s' % (utils.ctime(), sess, subj))

            ppc_file = h5py.File(output_file, 'a')
            prog = sess + '/' + subj
            if prog in ppc_file:
                continue  # Check if it's been run already

            data_file = h5py.File(phase_amp_data, 'r+')
            band_key = '%s/phase_data' % band
            subj_data = data_file[subj][sess][band_key][:, roi_indices]

            ppc = np.ndarray(shape=(len(rois), len(rois)))
            for r1, roi1 in enumerate(roi_indices):
                for r2, roi2 in enumerate(roi_indices):
                    if r1 == r2:
                        ppc[r1, r2] = 1
                    elif not r2 > r1:
                        phase_1 = subj_data[:, r1]
                        phase_2 = subj_data[:, r2]
                        rho = circ_corr(phase_1, phase_2)
                        ppc[r1, r2] = rho
                    else:
                        ppc[r1, r2] = 0

            out = ppc_file.require_group(prog)
            out.create_dataset('ppc', data=ppc, compression='lzf')
            ppc_file.close()
            data_file.close()
            del subj_data


def _calc_alpha_ppc():
    from misc import get_psd_rois
    psd_rois, _ = get_psd_rois()
    data_dir = utils.ProjectData.data_dir
    alpha_ppc = os.path.join(data_dir, 'MEG_alpha_ppc.hdf5')
    _calc_ppc(band='Alpha', rois=psd_rois, output_file=alpha_ppc)


def _calc_infraslow_ppc():
    from misc import get_psd_rois
    psd_rois, _ = get_psd_rois()
    data_dir = utils.ProjectData.data_dir
    alpha_ppc = os.path.join(data_dir, 'MEG_infraslow_ppc.hdf5')
    _calc_ppc(band='BOLD bandpass', rois=psd_rois, output_file=alpha_ppc)


if __name__ == "__main__":
    # _calc_alpha_ppc()
    _calc_infraslow_ppc()
