# -*- coding: UTF-8 -*-
"""Misc. work"""
import os
import utils
import ml_tools
import numpy as np
import pandas as pd


def get_psd_rois():
    ipsd_comp_file = './results/infraslow_PSD_model_comparison.xlsx'
    ipsd_comp = utils.load_xls(ipsd_comp_file)
    psd_rois, algorithm = ml_tools.pick_algorithm(ipsd_comp)
    return psd_rois, algorithm


def _pretty_pac(algorithm=None, rois=None):
    if not rois:
        rois = utils.ProjectData.glasser_rois
    pac = utils.load_phase_amp_coupling(rois=rois)

    manuscript_dir = './manuscript'
    fname = 'infraslow_phase_amp_coupling.xlsx'
    utils.save_xls(pac, os.path.join(manuscript_dir, fname))


def _pretty_infraslow_ppc(algorithm=None, rois=None):
    if not rois:
        rois = utils.ProjectData.glasser_rois
    ppc = utils.load_phase_phase_coupling(rois=rois)

    manuscript_dir = './manuscript'
    fname = 'infraslow_phase_phase_connectivity.xlsx'
    utils.save_xls(ppc, os.path.join(manuscript_dir, fname))


def _pretty_alpha_ppc(algorithm=None, rois=None):
    if not rois:
        rois = utils.ProjectData.glasser_rois
    data_dir = utils.ProjectData.data_dir
    ppc = utils.load_phase_phase_coupling(band='Alpha', rois=rois)

    manuscript_dir = './manuscript'
    fname = 'alpha_phase_phase_connectivity.xlsx'
    utils.save_xls(ppc, os.path.join(manuscript_dir, fname))


def _plot_brain(model='PSD', band='infraslow', alg='ExtraTrees', kernel=None):
    # Only works with ExtraTrees results
    results_dir = os.path.abspath('./results')
    feature_dir = utils.find_result(model, band, alg, kernel)
    feature_file = os.path.join(feature_dir, 'features.xlsx')
    feature_df = pd.read_excel(feature_file, index_col=0)
    rois = list(feature_df.index)

    importances = feature_df['Average Importance'].values
    brain_file = os.path.join(feature_dir, 'brain.nii.gz')
    brain = utils.create_custom_roi(rois, importances, brain_file)

    cmap = 'jet'
    figpath = os.path.join(feature_dir, 'brain.png')
    v = np.max(importances)
    utils.plot_brains(brain, maxval=v, cbar=True, cmap=cmap, figpath=figpath)


if __name__ == "__main__":
    # psd_rois, algorithm = get_psd_rois()
    # _pretty_pac(rois=psd_rois)
    # _pretty_infraslow_ppc(rois=psd_rois)
    # _pretty_alpha_ppc(rois=psd_rois)
    _plot_brain(band='alpha')
