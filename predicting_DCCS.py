# -*- coding: utf-8 -*-

"""Analysis script for the machine learning part of the project."""
import numpy as np
import pandas as pd
import utils
from copy import deepcopy
from os.path import join, abspath, dirname
from ml_tools import ML_pipeline, add_conjunction

# Global variables
glasser_rois = utils.ProjectData.glasser_rois
_, meg_sessions = utils.ProjectData.meg_metadata
card_sort_task_data = utils.load_behavior(behavior='CardSort_Unadj')


def _save_outputs(ML_pipe, output_dir):
    stack, conjunction = add_conjunction(ML_pipe.feature_importances)
    conjunction_results = {'Features': conjunction, 'Raw': stack}
    utils.save_xls(conjunction_results, join(output_dir, 'features.xlsx'))

    utils.save_xls(ML_pipe.predictions, join(output_dir, 'predictions.xlsx'))

    ML_pipe.model_performance.to_excel(
        join(output_dir, 'performance.xlsx'))


def perm_tests(ML_pipeline, n_iters):
    """Do permutation testing."""
    def _perm_df(n_iters, sessions):
        rownames = ['Iter %04d' % (n+1) for n in range(n_iters)]
        df = pd.DataFrame(index=rownames, columns=sessions)
        return df

    sessions = ML_pipeline.sessions
    expvar_perm = _perm_df(n_iters, sessions)
    mae_perm = _perm_df(n_iters, sessions)
    mse_perm = _perm_df(n_iters, sessions)

    if type(n_iters) != int:
        raise ValueError('Parameter "n_iters" must be an integer.')
    n = 0
    while n != n_iters:
        print('Permutation %03d: %s' % (n+1, ctime()))
        ML_pipeline.run_predictions(permute=n_iters)

        row = list(expvar_perm.index)[n]
        perm_perf = ML_pipeline.model_performance
        for s in sessions:
            expvar_perm.loc[row][s] = perm_perf.loc['ExplainedVariance'][s]
            mae_perm.loc[row][s] = perm_perf.loc['MAE'][s]
            mse_perm.loc[row][s] = perm_perf.loc['MSE'][s]

        n += 1
    perm_dict = {
        'ExplainedVariance': expvar_perm,
        'MSE': mse_perm,
        'MAE': mae_perm
        }
    return perm_dict


def _run_pipeline(
        predictors,
        targets,
        feature_names,
        session_names,
        random_state,
        permute,
        output_dir,
        reg_model='ExtraTrees',
        reg_kernel='linear'):
    """Pipeline convenience function."""
    feature_selection_grid = {
        'C': (.01, 1, 10, 100),
        "gamma": np.logspace(-2, 2, 5)
        }

    if reg_model == 'SVM':
        regression_grid = {
            'C': (.01, 1, 10, 100, 1000),
            "gamma": (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            }
    else:
        regression_grid = None

    ML_pipe = ML_pipeline(
        predictors=predictors,
        targets=targets,
        feature_selection_gridsearch=feature_selection_grid,
        model_gridsearch=regression_grid,
        feature_names=feature_names,
        session_names=session_names,
        random_state=random_state,
        debug=True)

    if not permute:
        ML_pipe.run_predictions(model=reg_model, model_kernel=reg_kernel)
        _save_outputs(ML_pipe, output_dir)
    else:
        ML_pipe.debug = False
        perm_dict = perm_tests(ML_pipe, n_iters=permute)
        utils.save_xls(perm_dict, join(output_dir, 'permutation_tests.xlsx'))


def _infraslow_psd_model(
        model, kernel, permute=False, seed=None, output_dir=None):
    """Predict DCCS using infraslow PSD."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))

    infraslow_psd = utils.load_infraslow_psd()
    _run_pipeline(
        predictors=infraslow_psd,
        targets=card_sort_task_data,
        feature_names=glasser_rois,
        session_names=meg_sessions,
        random_state=seed,
        permute=permute,
        output_dir=output_dir,
        reg_model=model,
        reg_kernel=kernel,
    )


def _infraslow_pac_model(
        model, kernel, rois=None, permute=False, seed=None, output_dir=None):
    """Predict DCCS using infraslow-alpha phase-amplitude coupling."""
    if not output_dir:
        output_dir = os.path.abspath(os.path.dirname(__file__))
    if not rois:
        rois = glasser_rois

    infraslow_pac = utils.load_phase_amp_coupling(rois=rois)
    _run_pipeline(
        predictors=infraslow_pac,
        targets=card_sort_task_data,
        feature_names=rois,
        session_names=meg_sessions,
        random_state=seed,
        permute=permute,
        output_dir=output_dir
    )


if __name__ == "__main__":
    from os import mkdir
    from os.path import isdir

    seed = 13  # For reproducibility

    print('Running ML with infraslow PSD: %s' % utils.ctime())
    ml_algorithms = ['ExtraTrees', 'SVM']
    kernels = ['linear', 'rbf']
    for m in ml_algorithms:
        if m == 'ExtraTrees':
            output_dir = "./analysis/infraslow_PSD_%s" % m
            if not isdir(output_dir):
                mkdir(output_dir)
            _infraslow_psd_model(
                model=m, kernel=None, seed=seed, output_dir=output_dir)
            # _infraslow_psd_model(
            #     model=m, kernel=None, permute=300, output_dir=output_dir)

        elif m == 'SVM':
            for k in kernels:
                output_dir = "./analysis/infraslow_PSD_%s_%s" % (m, k)
                if not isdir(output_dir):
                    mkdir(output_dir)
                _infraslow_psd_model(
                    model=m, kernel=k, seed=seed, output_dir=output_dir)
                # _infraslow_psd_model(
                #     model=m, kernel=k, permute=300, output_dir=output_dir)

    # _infraslow_psd_model(model=m, kernel=k, permute=300, output_dir=odir)

    # print('Running ML with infraslow PAC: %s' % utils.ctime())
    # odir = "./analysis/infraslow_PAC"
    # if not isdir(odir):
    #     mkdir(odir)
    # m = 'SVM'
    # k = 'linear'
    # _infraslow_pac_model(model=m, kernel=k, seed=seed, output_dir=odir)
    # _infraslow_pac_model(model=m, kernel=k, permute=300, output_dir=odir)
