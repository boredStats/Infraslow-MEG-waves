# -*- coding: utf-8 -*-

"""Running ML algorithms on infraslow MEG PSD and DCCS."""
import os
import utils
import ml_tools
import numpy as np
import pandas as pd
from copy import deepcopy

# Global variables
glasser_rois = utils.ProjectData.glasser_rois
_, meg_sessions = utils.ProjectData.meg_metadata
card_sort_task_data = utils.load_behavior(behavior='CardSort_Unadj')


def _infraslow_psd_model(
        alg, kernel, permute=False, seed=None, output_dir=None):
    """Predict DCCS using infraslow PSD."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))

    infraslow_psd = utils.load_infraslow_psd()

    feature_selection_grid = {
        'C': (.01, 1, 10, 100),
        "gamma": np.logspace(-2, 2, 5)
        }

    regression_grid = None
    if alg == 'SVM':
        regression_grid = {
            'C': (.01, 1, 10, 100, 1000),
            "gamma": (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            'degree': (2, 3, 4, 5)
            }

    ML_pipe = ml_tools.ML_pipeline(
        predictors=infraslow_psd,
        targets=card_sort_task_data,
        feature_selection_gridsearch=feature_selection_grid,
        model_gridsearch=regression_grid,
        feature_names=glasser_rois,
        session_names=meg_sessions,
        random_state=seed,
        debug=True)

    ML_pipe.run_predictions(model=alg, model_kernel=kernel)
    if not permute:
        ml_tools.save_outputs(ML_pipe, output_dir)
    else:
        ML_pipe.debug = False
        perm_dict = ml_tools.perm_tests(ML_pipe, n_iters=permute)
        utils.save_xls(
            perm_dict, os.path.join(output_dir, 'permutation_tests.xlsx'))


def _infraslow_pac_model(
        alg, kernel, permute=False, seed=None, output_dir=None, rois=None):
    """Predict DCCS using infraslow PAC."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))
    if not rois:
        rois = glasser_rois

    infraslow_pac = utils.load_phase_amp_coupling(rois=rois)

    feature_selection_grid = {
        'C': (.01, 1, 10, 100),
        "gamma": np.logspace(-2, 2, 5)
        }

    regression_grid = None
    if alg == 'SVM':
        regression_grid = {
            'C': (.01, 1, 10, 100, 1000),
            "gamma": (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            'degree': (2, 3, 4, 5)
            }

    ML_pipe = ml_tools.ML_pipeline(
        predictors=infraslow_pac,
        targets=card_sort_task_data,
        run_PLSC=True,
        feature_selection_gridsearch=feature_selection_grid,
        model_gridsearch=regression_grid,
        feature_names=glasser_rois,
        session_names=meg_sessions,
        random_state=seed,
        debug=True)

    ML_pipe.run_predictions(model=alg, model_kernel=kernel)
    if not permute:
        ml_tools.save_outputs(ML_pipe, output_dir)
    else:
        ML_pipe.debug = False
        perm_dict = ml_tools.perm_tests(ML_pipe, n_iters=permute)
        utils.save_xls(
            perm_dict, os.path.join(output_dir, 'permutation_tests.xlsx'))


def _infraslow_ppc_model(
        alg, kernel, permute=False, seed=None, output_dir=None, rois=None):
    """Predict DCCS using infraslow PPC."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))
    if not rois:
        rois = glasser_rois

    infraslow_pac = utils.load_phase_phase_coupling(rois=rois)

    feature_selection_grid = {
        'C': (.01, 1, 10, 100),
        "gamma": np.logspace(-2, 2, 5)
        }

    regression_grid = None
    if alg == 'SVM':
        regression_grid = {
            'C': (.01, 1, 10, 100, 1000),
            "gamma": (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            'degree': (2, 3, 4, 5)
            }

    ML_pipe = ml_tools.ML_pipeline(
        predictors=infraslow_pac,
        targets=card_sort_task_data,
        run_PLSC=True,
        feature_selection_gridsearch=feature_selection_grid,
        model_gridsearch=regression_grid,
        feature_names=glasser_rois,
        session_names=meg_sessions,
        random_state=seed,
        debug=True)

    ML_pipe.run_predictions(model=alg, model_kernel=kernel)
    if not permute:
        ml_tools.save_outputs(ML_pipe, output_dir)
    else:
        ML_pipe.debug = False
        perm_dict = ml_tools.perm_tests(ML_pipe, n_iters=permute)
        utils.save_xls(
            perm_dict, os.path.join(output_dir, 'permutation_tests.xlsx'))


def try_algorithms_on_psd():
    seed = 13  # For reproducibility
    print('Running ML with infraslow PSD: %s' % utils.ctime())
    ml_algorithms = ['ExtraTrees', 'SVM']
    kernels = ['linear', 'rbf', 'poly']  # only applies to SVM
    for m in ml_algorithms:
        if m == 'ExtraTrees':
            output_dir = "./results/infraslow_PSD_%s" % m
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            _infraslow_psd_model(
                alg=m, kernel=None, seed=seed, output_dir=output_dir)

        elif m == 'SVM':
            for k in kernels:
                output_dir = "./results/infraslow_PSD_%s_%s" % (m, k)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                _infraslow_psd_model(
                    alg=m, kernel=k, seed=seed, output_dir=output_dir)


def try_algorithms_on_pac(rois=None):
    seed = 13
    print('Running ML with infraslow PAC: %s' % utils.ctime())
    ml_algorithms = ['ExtraTrees', 'SVM']
    kernels = ['linear', 'rbf', 'poly']  # only applies to SVM
    for m in ml_algorithms:
        if m == 'ExtraTrees':
            output_dir = "./results/infraslow_PAC_%s" % m
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            _infraslow_pac_model(
                alg=m, kernel=None, seed=seed, output_dir=output_dir)

        elif m == 'SVM':
            for k in kernels:
                output_dir = "./results/infraslow_PAC_%s_%s" % (m, k)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                _infraslow_pac_model(
                    alg=m, kernel=k, seed=seed, output_dir=output_dir)


def try_algorithms_on_ppc(rois=None):
    seed = 13
    print('Running ML with infraslow PPC: %s' % utils.ctime())
    ml_algorithms = ['ExtraTrees', 'SVM']
    kernels = ['linear', 'rbf', 'poly']  # only applies to SVM
    for m in ml_algorithms:
        if m == 'ExtraTrees':
            output_dir = "./results/infraslow_PPC_%s" % m
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            _infraslow_pac_model(
                alg=m, kernel=None, seed=seed, output_dir=output_dir)

        elif m == 'SVM':
            for k in kernels:
                output_dir = "./results/infraslow_PPC_%s_%s" % (m, k)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                _infraslow_pac_model(
                    alg=m, kernel=k, seed=seed, output_dir=output_dir)


def main():
    # try_algorithms_on_psd()
    compare_dict = ml_tools.compare_algorithms()
    utils.save_xls(compare_dict, './results/model_comparison_PSD.xlsx')
    psd_rois = ml_tools.pick_algorithm(compare_dict)
    print(psd_rois)

    try_algorithms_on_pac(rois=psd_rois)
    compare_dict = ml_tools.compare_algorithms(model='PAC')
    utils.save_xls(compare_dict, './results/model_comparison_PAC.xlsx')

    try_algorithms_on_ppc(rois=psd_rois)
    compare_dict = ml_tools.compare_algorithms(model='PPC')
    utils.save_xls(compare_dict, './results/model_comparison_PPC.xlsx')


main()
