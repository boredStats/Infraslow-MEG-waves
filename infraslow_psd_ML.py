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
        model, kernel, permute=False, seed=None, output_dir=None):
    """Predict DCCS using infraslow PSD."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))

    infraslow_psd = utils.load_infraslow_psd()

    feature_selection_grid = {
        'C': (.01, 1, 10, 100),
        "gamma": np.logspace(-2, 2, 5)
        }

    regression_grid = None
    if model == 'SVM':
        regression_grid = {
            'C': (.01, 1, 10, 100, 1000),
            "gamma": (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
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

    if not permute:
        ML_pipe.run_predictions(model=model, model_kernel=kernel)
        ml_tools.save_outputs(ML_pipe, output_dir)
    else:
        ML_pipe.debug = False
        perm_dict = ml_tools.perm_tests(ML_pipe, n_iters=permute)
        utils.save_xls(
            perm_dict, os.path.join(output_dir, 'permutation_tests.xlsx'))


def try_algorithms():
    seed = 13  # For reproducibility
    print('Running ML with infraslow PSD: %s' % utils.ctime())
    ml_algorithms = ['ExtraTrees', 'SVM']
    kernels = ['linear', 'rbf']  # only applies to SVM
    for m in ml_algorithms:
        if m == 'ExtraTrees':
            output_dir = "./analysis/infraslow_PSD_%s" % m
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            _infraslow_psd_model(
                model=m, kernel=None, seed=seed, output_dir=output_dir)

        elif m == 'SVM':
            for k in kernels:
                output_dir = "./analysis/infraslow_PSD_%s_%s" % (m, k)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                _infraslow_psd_model(
                    model=m, kernel=k, seed=seed, output_dir=output_dir)


def main():
    # try_algorithms()
    compare_dict = ml_tools.compare_models(str_check='PSD')
    utils.save_xls(compare_dict, './analysis/model_comparison_PSD.xlsx')


main()
