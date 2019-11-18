# -*- coding: UTF-8 -*-
"""Misc. work"""
import os
import utils
import ml_tools


def get_psd_rois():
    ipsd_comp_file = './results/infraslow_PSD_model_comparison.xlsx'
    ipsd_comp = utils.load_xls(ipsd_comp_file)
    psd_rois, algorithm = ml_tools.pick_algorithm(ipsd_comp)
    return psd_rois, algorithm


def _pretty_pac(algorithm, rois=None):
    if not rois:
        rois = utils.ProjectData.glasser_rois
    pac = utils.load_phase_amp_coupling(rois=rois)

    manuscript_dir = './manuscript'
    fname = 'infraslow_phase_amp_coupling.xlsx'
    utils.save_xls(pac, os.path.join(manuscript_dir, fname))


def _pretty_ppc(algorithm, rois=None):
    if not rois:
        rois = utils.ProjectData.glasser_rois
    ppc = utils.load_phase_phase_coupling(rois=rois)

    manuscript_dir = './manuscript'
    fname = 'infraslow_phase_phase_connectivity.xlsx'
    utils.save_xls(ppc, os.path.join(manuscript_dir, fname))


def main():
    psd_rois, algorithm = get_psd_rois()
    _pretty_pac(algorithm, rois=psd_rois)
    _pretty_ppc(algorithm, rois=psd_rois)


main()
