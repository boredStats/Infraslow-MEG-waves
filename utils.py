# -*- coding: utf-8 -*-

"""Utility functions for this project."""
import os
import csv
import time
import h5py
import datetime
import numpy as np
import pandas as pd
import pickle as pkl


def ctime():
    """Return formatted time for performance testing."""
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def save_xls(dict_df, path):
    """Turn dictionary of dataframes to excel sheets.

    Save a dictionary of dataframes to an excel file with each dataframe
    as a seperate page.
    """
    writer = pd.ExcelWriter(path)
    for key in list(dict_df):
        dict_df[key].to_excel(writer, '%s' % key)

    writer.save()


def load_xls(path):
    """Turn a multi-sheet excel file into a dictionary."""
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    output = {}
    for sheet in sheets:
        output[sheet] = xls.parse(sheet_name=sheet, index_col=0)
    return output


def create_custom_roi(rois_to_combine, roi_magnitudes, fname=None):
    """Create a custom ROI based on ROI magnitudes.

    Uses Glasser ROI nifti files and a list of magnitudes to generate
    a single nifti file.
    """
    from nibabel import load, save, Nifti1Image
    data_dir = get_data_dir()
    roi_path = join(data_dir, 'glasser_atlas')

    def _stack_3d_dynamic(template, roi_indices, mag):
        t_copy = deepcopy(template)
        for num_counter in range(len(roi_indices[0])):
            x = roi_indices[0][num_counter]
            y = roi_indices[1][num_counter]
            z = roi_indices[2][num_counter]
            t_copy[x, y, z] = mag
        return t_copy

    rn = '%s.nii.gz' % rois_to_combine[0]
    t_vol = load(join(roi_path, rn))
    temp = t_vol.get_data()
    temp[temp > 0] = 0
    for r, roi in enumerate(rois_to_combine):
        if roi_magnitudes[r] == 0:
            pass
        rn = '%s.nii.gz' % roi
        volume_data = load(join(roi_path, rn)).get_data()
        r_idx = np.where(volume_data > 0)
        temp = _stack_3d_dynamic(temp, r_idx, roi_magnitudes[r])

    nifti = Nifti1Image(temp, t_vol.affine, t_vol.header)
    if fname is not None:
        save(nifti, fname)
    return nifti


def plot_brains(custom_roi, minval=0, maxval=None,
                figpath=None, cbar=False, cmap=None):
    """Plot inflated brains using nilearn plotting."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from nilearn import surface, plotting, datasets
    mpl.rcParams.update(mpl.rcParamsDefault)
    if cmap is None:
        cmap = 'coolwarm'

    fsaverage = datasets.fetch_surf_fsaverage()

    orders = [
        ('medial', 'left'),
        ('medial', 'right'),
        ('lateral', 'left'),
        ('lateral', 'right')]

    fig, ax = plt.subplots(
        2, 2,
        figsize=(8.0, 6.0),
        dpi=300, subplot_kw={'projection': '3d'})

    fig.subplots_adjust(hspace=0., wspace=0.00005)
    axes_list = fig.axes
    cbar_check = False
    for index, order in enumerate(orders):
        if index == len(orders)-1 and cbar is True:
            cbar_check = True
        view = order[0]
        hemi = order[1]

        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        plotting.plot_surf_roi(
            surf_mesh=fsaverage['infl_%s' % hemi],
            roi_map=texture,
            bg_map=fsaverage['sulc_%s' % hemi],
            cmap=cmap,
            hemi=hemi, view=view, bg_on_data=True,
            axes=axes_list[index],
            vmin=minval, vmax=maxval,
            output_file=figpath,
            symmetric_cbar=False,
            figure=fig,
            darkness=1,
            colorbar=cbar_check)

    plt.clf()


def load_behavior(behavior='CardSort_Unadj'):
    """Load a behavioral variable or all behavioral data."""
    data_dir = _get_data_dir()
    f = os.path.join(data_dir, 'behavioral_data.xlsx')
    behavior_df = pd.read_excel(f, index_col=0, sheet_name='cleaned')
    if behavior == 'all':
        return behavior_df
    else:
        try:
            behavior_series = behavior_df[behavior].astype(float)
            return behavior_series
        except ValueError:
            print('Behavior variable: %s was not found' % behavior)


def load_alpha_psd():
    data_dir = _get_data_dir()
    _, meg_sess = _get_meg_metadata()
    alpha_file = os.path.join(data_dir, 'MEG_alpha_power.xlsx')
    alpha_data = {}
    for sess in meg_sess:
        df = pd.read_excel(alpha_file, index_col=0, sheet_name=sess)
        alpha_data[sess] = df
    return alpha_data


def load_infraslow_psd(seperate_sessions=True):
    """Load pre-calculated infraslow PSD (.01 - .1 Hz bandpass)."""
    data_dir = _get_data_dir()
    rois = _get_glasser_rois(data_dir)

    power_df = pd.read_excel(
        os.path.join(data_dir, 'MEG_infraslow_power_methods.xlsx'),
        index_col=0, sheet_name='location')

    if seperate_sessions:
        session_dict = {}
        for session in ['Session1', 'Session2', 'Session3']:
            session_colums = [c for c in list(power_df) if session in c]

            session_df = power_df.loc[:, session_colums]
            column_dict = dict(zip(session_colums, rois))
            session_df.rename(columns=column_dict, inplace=True)
            session_dict[session] = session_df
        return session_dict
    else:
        return power_df


def load_phase_amp_coupling(phase_index=0, amp_index=3, rois=None):
    """Load PAC data.

    Defaults to BOLD-bandpass phase with Alpha amplitude.
    See freq_bands in ProjectData for more indices.
    """
    data_dir = _get_data_dir()
    file = os.path.join(data_dir, 'MEG_phase_amp_coupling.hdf5')

    meg_subj, sessions = _get_meg_metadata()
    if rois is None:
        rois = _get_glasser_rois()

    pac_dict = {}
    for sess in sessions:
        session_df = pd.DataFrame(index=meg_subj, columns=rois)
        for roi in rois:
            h5_file = h5py.File(file, 'r+')
            for subj in meg_subj:
                key = sess + '/' + subj + '/' + roi + '/r_vals'
                dset = h5_file[key][...]
                session_df.loc[subj][roi] = dset[phase_index, amp_index]
            h5_file.close()
        pac_dict[sess] = session_df
    return pac_dict


def load_phase_phase_coupling(file=None, rois=None):
    """Load PPC data."""
    data_dir = _get_data_dir()
    if file is None:
        # file = os.path.join(data_dir, 'MEG_phase_phase_coupling.hdf5')
        file = os.path.join(data_dir, 'psd_model_connectivity.hdf5')

    meg_subj, sessions = _get_meg_metadata()
    if rois is None:
        rois = _get_glasser_rois()
    glasser_rois = _get_glasser_rois()

    roi_indices = []
    for r in rois:
        roi_indices.append(glasser_rois.index(r))
    roi_indices = sorted(roi_indices)
    sorted_roi_names = []
    for r, roi in enumerate(roi_indices):
        sorted_roi_names.append(glasser_rois[roi])
    colnames = []
    for r1 in sorted_roi_names:
        for r2 in sorted_roi_names:
            connection = '%s %s' % (r1, r2)
            colnames.append(connection)

    session_dict = {}
    for sess in sessions:
        connection_array = np.ndarray(shape=(len(meg_subj), len(colnames)))
        for s, subj in enumerate(meg_subj):
            f = h5py.File(file, 'r')
            level = f[sess][subj]['ppc'][...]
            conn_counter = 0
            for r1 in range(len(sorted_roi_names)):
                for r2 in range(len(sorted_roi_names)):
                    val = level[r1, r2]
                    connection_array[s, conn_counter] = val
                    conn_counter += 1

            f.close()
        connection_df = pd.DataFrame(
            connection_array,
            index=meg_subj,
            columns=colnames)
        session_dict[sess] = connection_df

    return session_dict


def _get_data_dir():
    cdir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cdir, "data")


def _get_glasser_rois(data_dir=None):
    """Get Glasser atlas ROI labels."""
    if data_dir is None:
        data_dir = _get_data_dir()
    with open(os.path.join(data_dir, 'GlasserROIs.txt'), 'r') as file:
        rois = file.read().splitlines()
    return rois


def _get_meg_metadata(data_dir=None, bad_meg_subj=None):
    if data_dir is None:
        data_dir = _get_data_dir()
    if bad_meg_subj is None:
        bad_meg_subj = ['169040', '662551']

    with open(os.path.join(data_dir, 'proj_metatdata.pkl'), 'rb') as file:
        metadata = pkl.load(file)
    meg_subj = metadata['meg_subj']
    meg_sess = metadata['meg_sess']
    for b in bad_meg_subj:
        if b in meg_subj:
            meg_subj.remove(b)

    return meg_subj, meg_sess


def _get_mri_metadata(
        data_dir=None,
        bad_mri_subj=None):
    if data_dir is None:
        data_dir = _get_data_dir()
    if bad_mri_subj is None:
        bad_mri_subj = [
            '104012', '125525',
            '151526', '182840',
            '200109', '500222'
            ]

    with open(os.path.join(data_dir, 'proj_metatdata.pkl'), 'rb') as file:
        metadata = pkl.load(file)

    mri_subj = metadata['mri_subj']
    mri_sess = metadata['mri_sess']
    for b in bad_mri_subj:
        if b in mri_subj:
            mri_subj.remove(b)

    return mri_subj, mri_sess


def _proj_freq_bands():
    """Get frequency band names and ranges."""
    bands = {
        'BOLD bandpass': (.01, .1),  # Our infraslow range
        'Delta': (1.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 55)}
    return bands


def _get_gcolors(normalize=True):
    """Get Google20 colors (RGB triplets)."""
    data_dir = _get_data_dir()
    color_array = np.ndarray(shape=(20, 3))
    with open(os.path.join(data_dir, 'rgb_google20c.txt'), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for r, row in enumerate(reader):
            color = np.array([np.float(v) for v in row])
            if normalize:
                color = np.add(color, 1) / 256
            color_array[r, :] = color

    return color_array


class ProjectData:
    """HCP metadata and other important data for this project."""
    data_dir = _get_data_dir()
    glasser_rois = _get_glasser_rois(data_dir)
    meg_metadata = _get_meg_metadata(data_dir)
    mri_metadata = _get_mri_metadata(data_dir)
    gcolors = _get_gcolors()
    freq_bands = _proj_freq_bands()
