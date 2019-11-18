"""Plot predictions on lineplots."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import proj_utils as pu
import matplotlib.pyplot as plt
from math import hypot, sqrt
from os.path import join, abspath
from sympy import symbols, Eq, solve
from matplotlib.offsetbox import AnchoredText

output_dir = abspath('./../results/card_sort_regression')

proj_data = pu.ProjectData()
all_rois = proj_data.glasser_rois()
_, sessions = proj_data.meg_metadata()
session_plotnames = ['Session 1', 'Session 2', 'Session 3']


def create_coord_tuples(x, y):
    coords = []
    for n in range(len(x)):
        c = (x[n], y[n])
        coords.append(c)
    return coords


def unzip_coord_tuples(coords):
    x, y = [], []
    for c in coords:
        x.append(c[0])
        y.append(c[1])
    return x, y


def line(x, slope=1, intercept=0):
    return slope*x + intercept


def calculate_parallel(distance=1, side='top'):
    def _get_surround_coords(point, dist):
        x = point[0]
        y = point[1]
        top = (x-dist, y+dist)
        bottom = (x+dist, y-dist)
        return [top, bottom]

    x = np.linspace(0, 200)
    slope, intercept = 1, 0
    y = line(x, slope, intercept)

    line1 = create_coord_tuples(x, y)

    line2, line3 = [], []
    for c1 in line1:
        c2, c3 = _get_surround_coords(c1, dist=distance)
        line2.append(c2)
        line3.append(c3)

    x_a, y_a = unzip_coord_tuples(line2)
    x_b, y_b = unzip_coord_tuples(line3)
    if side == 'top':
        return x_a, y_a
    elif side == 'bottom':
        return x_b, y_b


def prediction_plot(
        predictions=None,
        error=None,
        r2=None,
        ax=None,
        pval=None,
        draw_xlabel=True,
        draw_ylabel=True,
        draw_legend=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 200)
    y = line(x)
    ax.plot(x, y, '-k')
    ax.set_xlim(80, 150)
    ax.set_ylim(80, 150)

    x_a, y_a = calculate_parallel(distance=error, side='top')
    ax.plot(x_a, y_a, '--k')
    x_b, y_b = calculate_parallel(distance=error, side='bottom')
    ax.plot(x_b, y_b, '--k')

    true = predictions[:, 0]
    pred = predictions[:, 1]
    ax.plot(true, pred, 'ob', fillstyle='none')

    if r2 is not None:
        if not pval or pval > .01:
            r2_text = r'$r^2 = %.2f$' % r2
        elif pval < .01:
            r2_text = r'$r^2 = %.2f$*' % r2
        elif pval < .001:
            r2_text = r'$r^2 = %.2f$**' % r2
        ax.annotate(r2_text, xy=(136.8, 80.8), xycoords='data', size=10)

    if draw_xlabel:
        ax.set_xlabel('True Scores', size='xx-large')
    if draw_ylabel:
        ax.set_ylabel('Predicted Scores', size='xx-large')
    if draw_legend:
        ax.legend(['Perfect prediction', 'RMSE'])

    return ax


def create_prediction_subplots(
        predictions,
        performance,
        err='MSE',
        subplot_titles=None,
        permfile=None):

    if type(performance) == pd.DataFrame:
        model_performance = performance
    else:
        model_performance = pd.read_excel(performance, index_col=0)

    try:
        perm_df = pd.read_excel(
            permfile, index_col=0, sheet_name='ExplainedVariance')
    except FileNotFoundError:
        perm_df = None

    sns.set()
    if subplot_titles is None:
        session_names = ['Session 1', 'Session 2', 'Session 3']
        subplot_titles = session_names

    ylab = [True, False, False]
    xlab = [False, True, False]
    fig, axes = plt.subplots(1, len(subplot_titles), figsize=(18, 6))
    p_values = []
    for s, subplot_title in enumerate(subplot_titles):
        ax = axes[s]

        if type(predictions) == dict:
            pred = predictions[subplot_title]
        else:
            pred = pd.read_excel(
                predictions,
                index_col=0,
                sheet_name=subplot_title)

        e = model_performance.loc[err][subplot_title]
        if err == 'MSE':
            error = np.sqrt(e)
        elif err == 'MAE':
            error = e

        r2 = model_performance.loc['ExplainedVariance'][subplot_title]
        if perm_df is not None:
            p = pu.permutation_p(r2, perm_df[subplot_title].values)
        else:
            p = None
        print(p)
        p_values.append(p)

        prediction_plot(
            pred.values, error=error, r2=r2, ax=ax, pval=p,
            draw_xlabel=xlab[s], draw_ylabel=ylab[s], draw_legend=ylab[s])
        ax.set_title(subplot_title, loc='left')

    return fig, p_values


def nice_perf_df(perf_file, p_values):
    performance_df = pd.read_excel(perf_file, index_col=0)
    rmse = np.sqrt(performance_df.loc['MSE'].values)
    r2 = performance_df.loc['ExplainedVariance'].values
    nice_df = pd.DataFrame(columns=list(performance_df))
    nice_df.loc['RMSE'] = rmse
    nice_df.loc['ExplainedVariance'] = r2
    nice_df.loc['p_values'] = p_values

    return nice_df


def plot_psd(output_dir, m='ExtraTreesRegressor'):
    psd_pred = join(output_dir, 'PSD_%s_predictions.xlsx' % m)
    psd_perf = join(output_dir, 'PSD_%s_performance.xlsx' % m)
    psd_perm = join(output_dir, 'PSD_perm_tests.xlsx')

    fig, p = create_prediction_subplots(
        psd_pred, psd_perf,
        permfile=psd_perm,
        # subplot_titles=['A1)', 'A2)', 'A3)'],
        subplot_titles=session_plotnames
        )
    fname = join(output_dir, 'PSD_%s_predictions.png' % m)
    fig.savefig(fname, bbox_inches='tight', dpi=300)

    nice_df = nice_perf_df(psd_perf, p)
    nice_df.to_excel(join(output_dir, 'PSD_%s_performance_cleaned.xlsx' % m))
    plt.close()


def plot_pac(m='SVR'):
    pac_pred = join(output_dir, 'PAC_%s_predictions.xlsx' % m)
    pac_perf = join(output_dir, 'PAC_%s_performance.xlsx' % m)
    pac_perm = join(output_dir, 'PAC_perm_tests.xlsx')
    fig, p = create_prediction_subplots(
        pac_pred, pac_perf,
        permfile=pac_perm,
        # subplot_titles=['B1)', 'B2)', 'B3)'],
        subplot_titles=session_plotnames)
    fname = join(output_dir, 'PAC_%s_predictions.png' % m)
    fig.savefig(fname, bbox_inches='tight', dpi=300)

    nice_df = nice_perf_df(pac_perf, p)
    nice_df.to_excel(join(output_dir, 'PAC_%s_performance_cleaned.xlsx' % m))
    plt.close()


def plot_ppc(m='SVR'):
    ppc_pred = join(output_dir, 'PPC_%s_predictions.xlsx' % m)
    ppc_perf = join(output_dir, 'PPC_%s_performance.xlsx' % m)
    ppc_perm = join(output_dir, 'PPC_perm_tests.xlsx')
    fig, p = create_prediction_subplots(
        ppc_pred, ppc_perf,
        permfile=ppc_perm,
        # subplot_titles=['C1)', 'C2)', 'C3)'],
        subplot_titles=session_plotnames
        )
    fname = join(output_dir, 'PPC_%s_predictions.png' % m)
    fig.savefig(fname, bbox_inches='tight', dpi=300)

    nice_df = nice_perf_df(ppc_perf, p)
    nice_df.to_excel(join(output_dir, 'PPC_%s_performance_cleaned.xlsx' % m))
    plt.close()


if __name__ == "__main__":
    calculate_parallel()
    plot_psd()
    plot_pac()
    plot_ppc()
