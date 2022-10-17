import os.path

import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

import seaborn as sns

from usable_functions_1 import *
from TiN_frame_process import dict_ind, str_descr, exp_descr, get_filtered_frame

from pyfitit import plotting

FIGSIZE_MAIN = (16 / 3 * 2, 9 / 3 * 2)
FIGSIZE_TALL = (16 / 3 * 2, 1.5 * 9 / 3 * 2)

FONTSIZE_MAIN = 15
FONTSIZE_1 = 14
FONTSIZE_2 = 6

MAIN_DPI = 400

matplotlib.rcParams.update({
        'mathtext.default': 'regular'
     })

EXT = '.jpeg'  # extention of the output files. variants: .png, .jpeg, .tiff

# because labels on the pictures changed bypassing me
LBLS_DICT = {
    'React_or_not': 'React',
    'Balanc_or_not': 'Balance',
    'ResidPress': '$P_{0}$',
    'ChambPress': '$P_{work}$',
    'N2ArRelation': 'N2/Ar',
    'VoltBias': 'Bias',
    'SubT': 'T',
    'DenCoat': 'CoatDen',
    'CoatComp': 'Ti/N',
    'Indent': 'Load',
    'CoatMu': 'COF',
}


def change_descrs_names_for_plot(frame_cols):
    li = frame_cols.to_list()
    for i, d in enumerate(li):
        if d in LBLS_DICT:
            li[i] = LBLS_DICT[d]
    return li


def save_csv(x, y, x_name, y_name, dest_path):
    df = pd.DataFrame(columns=[x_name, y_name])
    df[x_name] = x
    df[y_name] = y
    df.to_csv(dest_path, index=False)


# fig 2, S2
def count_sparsity_plotting_bar(frame, create_bar=False, out_file_path=f'./sparsity_picture{EXT}'):
    missing_mask = frame.isna().to_numpy()

    missing_counts_for_descrs = np.sum(missing_mask, axis=0)
    descrs_sparsity = missing_counts_for_descrs / frame.shape[0]

    missing_counts_for_exps = np.sum(missing_mask, axis=1)
    exps_sparsity = missing_counts_for_exps / frame.shape[1]

    if create_bar:
        fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
        ax.tick_params(axis='x', labelsize=FONTSIZE_1, labelrotation=90)
        ax.tick_params(axis='y', labelsize=FONTSIZE_1)
        fullness_arr = 1 - descrs_sparsity
        ids = np.argsort(fullness_arr)[::-1]

        bar_cols, bar_values = change_descrs_names_for_plot(frame.columns[ids]), fullness_arr[ids]
        file_with_no_ext, ext = os.path.splitext(out_file_path)
        save_csv(bar_cols, bar_cols, 'descr', 'fullness', f'{file_with_no_ext}.csv')
        ax.barh(bar_cols, bar_values,
               color=color_arr_from_arr(bar_values, init_color=(1, 0.9, 0.9), finish_color=(1, 0.2, 0.2), bottom=0., up=1.))
        ax.set_xlabel('1 - sparsity', fontsize=FONTSIZE_1)
        fig.savefig(out_file_path, dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)

    print('Sparsity of all data:\n', np.sum(missing_mask) / missing_mask.size)
    print('Sparsity per each descr:\n', descrs_sparsity)
    print('Sparsity per each descr, max:\n', np.max(descrs_sparsity))
    print('Sparsity per each descr, min:\n', np.min(descrs_sparsity))
    print('Sparsity per each descr, median:\n', np.median(descrs_sparsity))
    print('Sparsity per each exp:\n', exps_sparsity)
    print('Sparsity per each exp, max:\n', np.max(exps_sparsity))
    print('Sparsity per each exp, min:\n', np.min(exps_sparsity))
    print('Sparsity per each exp, median:\n', np.median(exps_sparsity))


# fig 3
def bar_for_get_result(data_file_path, bar_descrs, out_folder=None, out_file_name=None, add_text=None,
                       add_text_plot=None, text_plot_ops=None, one_more_file_path=None):
    # read and check
    r2_data = pd.read_excel(data_file_path)
    # r2_data.to_excel('bar_check.xlsx')
    other_r2 = one_more_r2_data = None
    if one_more_file_path is not None:
        one_more_r2_data = pd.read_excel(one_more_file_path)
    # imputers
    imp_arr = np.array(r2_data.loc[0, 0:])
    _, imp_inds = np.unique(imp_arr, return_index=True)
    imp_inds.sort()
    imp_mass = imp_arr[imp_inds]
    # models
    model_arr = np.array(r2_data.loc[1, 0:])
    _, model_inds = np.unique(model_arr, return_index=True)
    model_inds.sort()
    model_mass = model_arr[model_inds]
    # data
    num_imps = imp_mass.shape[0]
    num_models = model_mass.shape[0]
    num_cols = num_imps * num_models
    x_data = np.arange(num_cols)
    print(x_data)
    x_data = x_data.astype('float64')
    for i in range(0, num_cols, num_models):
        x_data[i] = (3 * x_data[i] + x_data[i + 1]) / 4
        x_data[i + 2] = (3 * x_data[i + 2] + x_data[i + 1]) / 4
    print(x_data)
    colors = color_arr_from_arr(np.arange(num_models), init_color=(0.5, 0.7, 0.95), finish_color=(1, 0.4, 0.4), bottom=0, up=(num_models + 1))

    # when textplot labels for columns
    color_per_name = dict()
    if isinstance(add_text_plot, list):
        for i, name in enumerate(('ExtraTrees', 'SVM', 'RidgeCV')):
            color_per_name[name] = colors[i]

    print(colors)
    labels = []
    for i in range(num_cols):
        if i % num_models == num_models // 2:
            lbl = imp_mass[i // num_models]
            if lbl == 'knn':
                lbl = 'KNN'
            labels.append(f'{lbl}')
        else:
            labels.append('')
    width = 2.75 / np.max([num_cols - 8, 1])
    # rects_list = []
    # sgn = -1
    # build graphs
    for i, descr in enumerate(bar_descrs):
        fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
        # fig.update_layout(legend_orientation='h')
        r2_values = r2_data.loc[i + 2, 0:]
        r2_values[r2_values < 0.05] = 0.05
        if one_more_r2_data is not None:
            other_r2 = one_more_r2_data.loc[i + 2, 0:]
            other_r2[other_r2 < 0.05] = 0.05
        for i1, model in enumerate(model_mass):
            x_arr = x_data[num_models * np.arange(num_imps) + i1]
            r2_arr = r2_values[num_models * np.arange(num_imps) + i1]
            color = colors[i1]
            if one_more_r2_data is None:
                ax.bar(x_arr, r2_arr, width=width, label=model, color=color)
            else:
                other_r2_arr = other_r2[num_models * np.arange(num_imps) + i1]
                d_arr = abs(r2_arr - other_r2_arr)
                min_arr = other_r2_arr
                ids = other_r2_arr > r2_arr
                min_arr[ids] = r2_arr[ids]
                ax.bar(x_arr, min_arr, width=width, label=model, color=color, hatch='XX')
                ax.bar(x_arr[ids], d_arr[ids], bottom=min_arr[ids], width=width, label=model, color=color)
                ids = other_r2_arr < r2_arr
                ax.bar(x_arr[ids], d_arr[ids], bottom=min_arr[ids], width=width, label=model, color='white', hatch='XX')
        # rects_list.append(rects)
        # sgn *= -1
        if add_text_plot is not None:
            if text_plot_ops is None:
                text_plot_ops = dict()
            elif 'transform' in text_plot_ops:
                text_plot_ops['transform'] = ax.transAxes
            if isinstance(add_text_plot, list):
                for x_y_s in add_text_plot:
                    ax.text(*x_y_s, **text_plot_ops, color=color_per_name[x_y_s[2]], fontsize=FONTSIZE_MAIN)
            elif isinstance(add_text_plot, dict):
                ax.text(*add_text_plot, **text_plot_ops, fontsize=FONTSIZE_MAIN)
        ax.set_ylabel('R2 score', fontsize=FONTSIZE_MAIN)
        if add_text is not None:
            ax.set_title(f'R2 scores\n{add_text}', fontsize=FONTSIZE_MAIN)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(x_data)
        ax.set_xticklabels(labels)
        ax.tick_params(labelsize=FONTSIZE_1)
        # ax.legend()
        if out_file_name is None:
            out_file_name = f'score_bars_{descr}'
        if out_folder is None:
            out_folder = os.path.dirname(data_file_path)
        fig.savefig(f'{out_folder}/{out_file_name}{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


# fig4
def importance_bars(in_path, out_folder, name):
    # TODO: this function really depends on external effects, fix it
    importance_data = pd.read_excel(in_path)
    flag = False
    for col in importance_data.columns:
        if '_' not in col:
            flag = True
            del importance_data[col]
    if flag:
        importance_data.to_excel(in_path)
    imp_mass = ['const', 'simple', 'iterative', 'knn']
    columns = importance_data.columns.to_list()
    for i, imp in enumerate(imp_mass):
        assert columns[2 * i][columns[2 * i].rfind('_') + 1:] == imp
        assert columns[2 * i + 1][columns[2 * i + 1].rfind('_') + 1:] == imp, columns[i][columns[2 * i + 1].rfind('_') + 1:]
        fig, ax = plt.subplots(1, figsize=FIGSIZE_MAIN)
        ax.tick_params(labelsize=FONTSIZE_1)
        ax.barh(importance_data[columns[2 * i]], importance_data[columns[2 * i + 1]],
                color=color_arr_from_arr(np.array(importance_data[columns[2 * i + 1]]), init_color=(1.0, 1.0, 1.0), finish_color=(1.0, 0.9, 0.4), bottom=-0.25))
        # ax.barh(used_columns, model_name.feature_importances_, color='orange')
        # ax.tick_params(labelrotation=45)
        ax.set_title(f'Feature importance {name}', fontsize=FONTSIZE_MAIN)
        fig.savefig(f'{out_folder}/feature_importance_{imp}_{name}{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def bars_for_descr_analysis(folder=''):
    graph_data = pd.read_excel(folder + 'descr_analysis.xlsx')
    # graph_data.to_excel('bar_check.xlsx')
    columns = graph_data['descriptor']
    fig, ax = plt.subplots(1, figsize=FIGSIZE_MAIN)
    ax.tick_params(axis='x', labelrotation=90, labelsize=FONTSIZE_1)
    ax.tick_params(axis='y', labelsize=FONTSIZE_1)
    ax.bar(change_descrs_names_for_plot(columns), graph_data['r2_arr'], color=color_arr_from_arr(np.array(graph_data['r2_arr'])))
    plt.savefig(folder + f'score_all_by_all{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1, figsize=FIGSIZE_MAIN)
    ax.tick_params(axis='x', labelrotation=90, labelsize=FONTSIZE_1)
    ax.tick_params(axis='y', labelsize=FONTSIZE_1)
    ax.bar(change_descrs_names_for_plot(columns), graph_data['statistic'],
           color=color_arr_from_arr(np.array(graph_data['statistic']), init_color=(0.05, 0.05, 0.3),
                                    finish_color=(0.9, 0.8, 0.6), bottom=0., up=1.))
    fig.savefig(folder + f'statistic_{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def get_descr_distribution_picture(frame, descr, out_folder='', bins=50, color='#ffba54'):
    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
    data = np.array(frame.loc[dict_ind[descr], descr])
    total_values = data[data is not np.nan].size
    ax.hist(data, bins=bins, color=color, histtype='bar')
    ax.set_xlabel(f'Value of {descr}', fontsize=FONTSIZE_MAIN)
    ax.set_ylabel('Count', fontsize=FONTSIZE_MAIN)
    ax.set_title(f'{descr} distribution\ntotal number of values: {total_values}', fontsize=FONTSIZE_MAIN)
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.savefig(f'{out_folder}{descr}{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


# fig 1
def descr_sparcity_table(df, columns, descr='', all_table=False, out_folder='.'):
    if all_table:
        matr = df.T.isna().astype('int64').to_numpy()
        # title = 'График разреженности\nбазы данных'
    else:
        assert len(descr), 'expect descriptor`s name, received nothing'
        matr = df.loc[dict_ind[descr], columns].T.isna().astype('int64').to_numpy()
        # title = f'График, наглядно демонстрирующий разреженность матрицы данных составленной из столбцов в которых {descr} notNull'
    f, ax = plt.subplots(figsize=FIGSIZE_TALL)
    ax.tick_params(labelsize=FONTSIZE_1)
    ax = sns.heatmap(matr, linewidths=0.1, xticklabels=False, yticklabels=change_descrs_names_for_plot(columns),
                     ax=ax, cbar=False, cmap=['blue', 'white'])
    # ax.set_title(title, fontsize=MAIN_PLOT_FONT_SIZE)
    articles_number = np.unique(df['PaperID']).size
    ax.text(0.5, -0.05, f'{matr.shape[1]} experiments from {articles_number} articles',
            transform=ax.transAxes, va='center', ha='center', fontsize=FONTSIZE_1)
    # ax.arrow(0.01, 0., 0.95, 0., transform=ax.transAxes, width=0.005, head_length=4.5 * 0.005, shape='full',
    #          color=(0.3, 0.3, 0.3,), zorder=4)
    if all_table:
        f.savefig(f'{out_folder}/all_table_picture{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    else:
        f.savefig(f'{out_folder}/{descr}_picture{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(f)


# эта функция отчасти дублирует функцию get_scatter_plots
def descr_correlate_picture(frame, used_descr, target_descr, out_folder='', mode='many_pictures'):
    inds = np.array(intersect_lists(dict_ind[target_descr], dict_ind[used_descr]))
    if not inds.size:
        return
    if used_descr in str_descr:
        return
    data = frame.loc[inds, ['PaperID', used_descr, target_descr]]
    print(data.shape)
    if mode == 'many_pictures':
        for p in np.unique(data['PaperID']):
            if np.unique(data.loc[data.index[data['PaperID'] == p], used_descr]).size <= 3:
                data = data.loc[data.index[data['PaperID'] != p], :]
            else:
                fig, ax = plt.subplots(1, figsize=FIGSIZE_MAIN)
                ax.plot(data.loc[data.index[data['PaperID'] == p], used_descr], data.loc[data.index[data['PaperID'] == p], target_descr], '-o', c=np.random.choice(['red', 'green', 'blue', 'yellow', 'pink']))
                ax.set_xlabel(used_descr)
                ax.set_ylabel(target_descr)
                fig.savefig(f'{out_folder}{used_descr}_{target_descr}_paper{p}{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
                plt.close(fig)
    elif mode == 'one_picture':
        for p in np.unique(data['PaperID']):
            if np.std(data.loc[data.index[data['PaperID'] == p], used_descr]) < 1e-5:
                data = data.loc[data.index[data['PaperID'] != p], :]
        print(data.shape)
        if not data.shape[0]:
            return
        papers = np.array(data['PaperID'])
        color_data = np.zeros_like(papers)
        for i in range(papers.size):
            color_data[i] = int(papers[i][:-1])
        fig, ax = plt.subplots(1, figsize=FIGSIZE_MAIN)
        ax.scatter(data[used_descr], data[target_descr], c=color_data)
        ax.set_xlabel(used_descr)
        ax.set_ylabel(target_descr)
        fig.savefig(f'{out_folder}{used_descr}_{target_descr}_picture{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


# fig S1
def arts_descrs_picture(frame, out_folder='.', hist_or_bar='hist'):
    _, counts = np.unique(frame['PaperID'], return_counts=True)
    fig, ax = plt.subplots(1, figsize=FIGSIZE_TALL)
    if hist_or_bar == 'hist':
        _, frequency = np.unique(counts, return_counts=True)
        max_counts = np.max(counts)
        ax.hist(counts, bins=max_counts - 1)
        ax.set_xlabel('Number of experiments \nin article')
        ax.set_ylabel('Number of articles')
        ax.set_xticks(np.arange(max_counts + 1))
    elif hist_or_bar == 'bar':
        unique_counts, frequency = np.unique(counts, return_counts=True)
        save_csv(unique_counts, frequency, 'counts', 'frequency', f'{out_folder}/arts_descrs_picture.csv')
        ax.bar(unique_counts, frequency, color=(0.8, 0.8, 0.25))
        ax.tick_params(labelsize=FONTSIZE_1)
        ax.set_xlabel('Number of experiments in article', fontsize=16)
        ax.set_ylabel('Number of articles', fontsize=16)
    ax.set_yticks(np.arange(np.max(frequency) + 1))
    fig.savefig(f'{out_folder}/arts_descrs_picture{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def get_articles_picture(folder, score_mod='relative'):
    if score_mod == 'relative':
        vmin = -10
        vmax = 0
    elif score_mod == 'mean':
        vmin = 0
        vmax = 1
    elif score_mod == 'wide':
        vmin = -1
        vmax = 1
    else:
        assert False, f'invalid value for score_mod: {score_mod}'
    input_file = folder + 'ArticleAnalysis_' + score_mod + '.xlsx'
    columns = pd.read_excel(input_file, usecols='B')
    frame = pd.read_excel(input_file)
    print(frame.columns[2:])
    f, ax = plt.subplots(figsize=FIGSIZE_MAIN)
    ax = sns.heatmap(frame.to_numpy()[:, 2:].astype('float64'), linewidths=0.1, linecolor='black',
                     xticklabels=frame.columns[2:], yticklabels=list(columns['descr']),
                     ax=ax, cbar=True, vmax=vmax, vmin=vmin, cmap='summer_r')
    ax.tick_params(labelsize=FONTSIZE_1)
    f.tight_layout()
    f.savefig(folder + 'articles_picture_' + score_mod + f'{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(f)


def get_scatter_plots(frame, descr, list_of_descrs, out_folder='', mode='all'):
    for name in list_of_descrs:
        if mode == 'all':
            x_data = frame.loc[dict_ind[name], name]
            y_data = frame.loc[dict_ind[name], descr]
        arts = frame.loc[dict_ind[name], 'PaperID'].to_numpy()
        for i in range(arts.size):
            arts[i] = int(arts[i][:-1])
        arts = arts.astype('int32')
        plotting.scatter(x_data, y_data, color=arts, colorMap='gist_rainbow',
                         fileName=f'{out_folder}{name}_scatter{EXT}')


def quality_heatmap(in_folder='.', out_folder=''):
    df = pd.read_excel(f'{in_folder}/qheatmap_data.xlsx')
    del df['Unnamed: 0']
    matrix_r2 = df.to_numpy()
    features = df.columns
    f, ax = plt.subplots(1, figsize=(FIGSIZE_MAIN[0], 1.2 * FIGSIZE_MAIN[1]))
    ax = sns.heatmap(matrix_r2, xticklabels=features, vmin=0., vmax=0.8, yticklabels=features)
    ax.tick_params(axis='x', labelrotation=90, labelsize=FONTSIZE_1)
    ax.tick_params(axis='y', labelsize=FONTSIZE_1)
    # ax.set_xlabel('predicted')
    # ax.set_ylabel('true')
    f.savefig(f'{out_folder}/heat_map{EXT}', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(f)


def all_plots(frame, out_folder, **kwargs):

    def data_stat_block(df, subdir=None):
        if subdir is not None:
            plot_dir_path = f'{out_folder}/{subdir}'
        else:
            plot_dir_path = out_folder
        arts_descrs_picture(df, plot_dir_path, hist_or_bar='bar')
        count_sparsity_plotting_bar(df.loc[:, remove_many(df.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                                    out_file_path=f'{plot_dir_path}/fig2{EXT}')
        descr_sparcity_table(df, df.columns, all_table=True, out_folder=plot_dir_path)

    if 'filter_ops' in kwargs:
        os.makedirs(f'{out_folder}/unfiltered/', exist_ok=True)
        os.makedirs(f'{out_folder}/filtered/', exist_ok=True)
        data_stat_block(frame, 'unfiltered')
        frame = get_filtered_frame(frame, *(kwargs['filter_ops']))
        data_stat_block(frame, 'filtered')
    else:
        data_stat_block(frame)

    if 'results_ops' in kwargs:
        bar_for_get_result(out_folder=out_folder, **(kwargs['results_ops']))
    if 'importance_ops' in kwargs:
        importance_bars(out_folder=out_folder, **(kwargs['importance_ops']))
    if 'qheatmap_ops' in kwargs:
        quality_heatmap(*(kwargs['qheatmap_ops']), out_folder)


# def all_plots(main_path=''):
#     # arts descrs picture
#     frame = pd.read_excel(main_path + 'origin_frame.xlsx')
#     frame.index = pd.read_excel(main_path + 'origin_frame.xlsx', usecols='A')['Unnamed: 0']
#     del frame['Unnamed: 0']
#     arts_descrs_picture(frame, hist_or_bar='bar', out_folder=main_path)
#     current_path = main_path + 'descr_analysis/'
#     bars_for_descr_analysis(folder=current_path)
#     for fold in 'no_filter/', 'filter/':
#         current_path = main_path + fold
#         frame = pd.read_excel(current_path + 'x_file.xlsx')
#         frame.index = pd.read_excel(current_path + 'x_file.xlsx', usecols='A')['Unnamed: 0']
#         del frame['Unnamed: 0']
#         columns = frame.columns
#         # sparsity pictures
#         count_sparsity_plotting_bar(frame.loc[:, remove_many(columns, ['PaperID', 'Bad'])], create_bar=True, out_folder=current_path)
#         descr_sparcity_table(frame, columns, all_table=True, out_folder=current_path)
#         # article_analysis pictures
#         for mod in 'relative', 'mean':
#             get_articles_picture(current_path, score_mod=mod)
#         # get_result pictures
#         bar_descrs = ['H']
#         bar_for_get_result(current_path, bar_descrs=clean_list(frame, bar_descrs, inplace=False))
#     current_path = main_path + 'filter/'
#     # importance pictures
#     importance_bars(current_path, 'H')
#     # quality heatmap
#     quality_heatmap(
#         in_out_folder=current_path)
