# -*- coding: utf-8 -*-

"""Plot gridsearch results."""

import numpy as np
import pandas as pd
import proj_utils as pu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from os.path import isdir, join, abspath
from os import mkdir

proj_data = pu.ProjectData()
all_rois = proj_data.glasser_rois()
_, sessions = proj_data.meg_metadata()

output_dir = abspath('./../results/card_sort_regression')
if not isdir(output_dir):
    mkdir(output_dir)


def extract_grid_data(df):
    columns_to_grab = [
        'params',
        'param_C',
        # 'param_epsilon',
        'param_gamma',
        'mean_test_score',
        'rank_test_score'
        ]
    extract_df = df[columns_to_grab]
    params = [p for p in columns_to_grab if 'param_' in p]
    param_dict = {}
    for p in params:
        vals = extract_df[p].values
        param_dict[p] = np.unique(vals)
    return extract_df, param_dict


def create_plot_lines(grid_data, param_dict):
    C = param_dict['param_C']
    epsilon = param_dict['param_epsilon']
    gamma = param_dict['param_gamma']

    lines = {}
    for c in C:
        for e in epsilon:
            line_data = []
            for g in sorted(gamma):
                row = grid_data[(grid_data['param_C'] == c) &
                                (grid_data['param_epsilon'] == e) &
                                (grid_data['param_gamma'] == g)]
                score = row['mean_test_score'].values[-1]
                line_data.append(score)
            lines['%s,%s' % (c, e)] = line_data
    return lines, C, epsilon, sorted(gamma)


def create_plot_lines_noe(grid_data, param_dict):
    C = param_dict['param_C']
    gamma = param_dict['param_gamma']

    lines = {}
    for c in C:
        line_data = []
        for g in sorted(gamma):
            row = grid_data[(grid_data['param_C'] == c) &
                            (grid_data['param_gamma'] == g)]
            score = row['mean_test_score'].values[-1]
            line_data.append(score)
        lines['%s' % c] = line_data
    return lines, C, sorted(gamma)


def grid_plot(lines, C, epsilon, gamma, title=None, plot_legend=True):
    sns.set()
    line_colors = [
        'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan']
    line_styles = ['-.', '--', ':']
    marker_styles = ['o', 'v', 's']

    color_dict = dict(zip(C, line_colors))
    linestyle_dict = dict(zip(epsilon, line_styles))
    marker_dict = dict(zip(epsilon, marker_styles))

    fig, ax = plt.subplots(figsize=(12, 9))
    for param_combo in lines:
        str_split = param_combo.split(',')
        c = float(str_split[0])
        e = float(str_split[-1])

        color = color_dict[c]
        linestyle = linestyle_dict[e]
        marker = marker_dict[e]

        y = lines[param_combo]
        x = np.arange(len(y))
        ax.plot(
            x, y,
            marker='o',
            linestyle=linestyle,
            color=color,
            # marker=marker,
            markerfacecolor="None",
            linewidth=.8,
            markersize=5,
            )

    ax.set_xlabel('Gamma', size='x-large')
    plt.xticks(np.arange(len(y)), gamma, fontsize='medium')
    ax.set_ylabel(r'$r^2$', size='xx-large')
    ax.tick_params(axis=y, labelsize='medium')

    if plot_legend:
        current_ylim = ax.get_ylim()  # save current ylims (legend prep)

        legend_y = [-5 for n in range(len(x))]
        handles = []
        for e in linestyle_dict:
            legend_linestyle = linestyle_dict[e]
            legend_label = 'epsilon: %s' % e
            p = mpatches.Patch(
                linestyle=legend_linestyle,
                label=legend_label,
                linewidth=1,
                color='k',
                fill=False)
            handles.append(p)

        for c in color_dict:
            legend_color = color_dict[c]
            legend_label = 'C: %s' % c
            p = mpatches.Patch(
                color=legend_color,
                label=legend_label,
                linestyle='-',
                # fill=False,
                linewidth=1,
            )
            handles.append(p)
        plt.legend(handles=handles)

        ax.set_ylim(bottom=current_ylim[0], top=current_ylim[-1])

    if title is not None:
        plt.title(title)
    return fig, ax


def grid_plot_noe(lines, C, gamma, title=None, plot_legend=True):
    sns.set()
    line_colors = [
        'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan']

    color_dict = dict(zip(C, line_colors))

    fig, ax = plt.subplots(figsize=(12, 9))
    for c_str in lines:
        c = float(c_str)
        color = color_dict[c]

        y = lines[c_str]
        x = np.arange(len(y))
        ax.plot(
            x, y,
            marker='o',
            linestyle='-',
            color=color,
            markerfacecolor="None",
            linewidth=1,
            markersize=5,
            )

    ax.set_xlabel('Gamma', size='x-large')
    plt.xticks(np.arange(len(y)), gamma, fontsize='medium')
    ax.set_ylabel(r'Average $r^2$', size='xx-large')
    ax.tick_params(axis=y, labelsize='medium')

    if plot_legend:
        current_ylim = ax.get_ylim()  # save current ylims (legend prep)

        legend_y = [-5 for n in range(len(x))]
        handles = []

        for c in color_dict:
            legend_color = color_dict[c]
            legend_label = 'C: %s' % c
            p = mpatches.Patch(
                color=legend_color,
                label=legend_label,
                linestyle='-',
                # fill=False,
                linewidth=1,
            )
            handles.append(p)
        plt.legend(handles=handles)

        ax.set_ylim(bottom=current_ylim[0], top=current_ylim[-1])

    if title is not None:
        plt.title(title)
    return fig, ax


def create_nice_grid_df(raw_gridsearch):
    columns_to_grab = ['param_C', 'param_gamma', 'mean_test_score']
    new_colnames = ['C', 'gamma', 'Average r2']
    nice_gridsearch = raw_gridsearch[columns_to_grab]
    cleaned_gridsearch = nice_gridsearch.rename(
        columns=dict(zip(columns_to_grab, new_colnames)))
    return cleaned_gridsearch


def plot_pac_gridsearch():
    gridsearch_file = join(output_dir, 'PAC_SVR_gridsearch.xlsx')
    gridsearch_df = pd.read_excel(gridsearch_file, index_col=0)
    grid_data, params = extract_grid_data(gridsearch_df)

    title = 'Phase-amplitude coupling model - hyperparameter gridsearch'

    # lines, C, epsilon, gamma = create_plot_lines(grid_data, params)
    # fig, _ = grid_plot(lines, C, epsilon, gamma, title=title)

    lines, C, gamma = create_plot_lines_noe(grid_data, params)
    fig, _ = grid_plot_noe(lines, C, gamma, title=title)
    fig.savefig(
        join(output_dir, 'PAC_SVR_gridsearch.png'),
        bbox_inches='tight',
        dpi=300,)

    nice_grid = create_nice_grid_df(gridsearch_df)
    nice_grid.to_excel(join(output_dir, 'PAC_SVR_cleaned_gridsearch.xlsx'))


def plot_ppc_gridsearch():
    gridsearch_file = join(output_dir, 'PPC_SVR_gridsearch.xlsx')
    gridsearch_df = pd.read_excel(gridsearch_file, index_col=0)
    grid_data, params = extract_grid_data(gridsearch_df)

    title = 'Phase-phase connectivity model - hyperparameter gridsearch'

    # lines, C, epsilon, gamma = create_plot_lines(grid_data, params)
    # fig, _ = grid_plot(lines, C, epsilon, gamma, title=title)

    lines, C, gamma = create_plot_lines_noe(grid_data, params)
    fig, _ = grid_plot_noe(lines, C, gamma, title=title)
    fig.savefig(
        join(output_dir, 'PPC_SVR_gridsearch.png'),
        bbox_inches='tight',
        dpi=300,)

    nice_grid = create_nice_grid_df(gridsearch_df)
    nice_grid.to_excel(join(output_dir, 'PPC_SVR_cleaned_gridsearch.xlsx'))

def main():
    plot_pac_gridsearch()
    plot_ppc_gridsearch()


main()
