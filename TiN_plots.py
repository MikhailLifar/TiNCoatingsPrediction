import matplotlib.pyplot as plt
import \
    numpy as np
import seaborn as sns

from usable_functions_1 import *
from TiN_frame_process import dict_ind, str_descr, exp_descr

from pyfitit import plotting

MAIN_FIG_SIZE = (16 / 3 * 2, 9 / 3 * 2)
MAIN_PLOT_FONT_SIZE = 15
BIG_LABELSIZE = 14
SMALL_LABELSIZE = 6
MAIN_DPI = 300


def count_sparsity(frame, create_bar=False, out_folder='', out_file='sparsity_picture.png'):
    missing_mask = frame.isna().to_numpy()
    missing_counts_for_descrs = np.sum(missing_mask, axis=0)
    missing_counts_for_exps = np.sum(missing_mask, axis=1)
    descrs_sparsity = missing_counts_for_descrs / frame.shape[0]
    exps_sparsity = missing_counts_for_exps / frame.shape[1]
    if create_bar:
        fig, ax = plt.subplots(figsize=MAIN_FIG_SIZE)
        ax.tick_params(axis='x', labelsize=BIG_LABELSIZE, labelrotation=90)
        ax.tick_params(axis='y', labelsize=BIG_LABELSIZE)
        ax.bar(frame.columns, 1 - descrs_sparsity,
               color=color_arr_from_arr(1 - descrs_sparsity, init_color=(1, 0.9, 0.9), finish_color=(1, 0.2, 0.2), bottom=0., up=1.))
        fig.savefig(out_folder + out_file, dpi=MAIN_DPI, bbox_inches='tight')
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


def bar_for_get_result(filepath, bar_descrs, out_file_name=None, add_text=None):
    # read and check
    r2_data = pd.read_excel(filepath)
    # r2_data.to_excel('bar_check.xlsx')
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
    print(colors)
    labels = []
    for i in range(num_cols):
        if i % num_models == num_models // 2:
            labels.append(f'{imp_mass[i // num_models]}')
        else:
            labels.append('')
    width = 2.75 / np.max([num_cols - 8, 1])
    # rects_list = []
    # sgn = -1
    # build graphs
    for i, descr in enumerate(bar_descrs):
        fig, ax = plt.subplots(figsize=MAIN_FIG_SIZE)
        # fig.update_layout(legend_orientation='h')
        r2_values = r2_data.loc[i + 2, 0:]
        r2_values[r2_values < 0.05] = 0.05
        for i1, model in enumerate(model_mass):
            ax.bar(x_data[num_models * np.arange(num_imps) + i1], r2_values[num_models * np.arange(num_imps) + i1], width=width, label=model, color=colors[i1])
        # rects_list.append(rects)
        # sgn *= -1
        # if add_text_plot is not None:
        #     ax.text(**add_text_plot, fontsize=MAIN_PLOT_FONT_SIZE)
        ax.set_ylabel('R2 score', fontsize=MAIN_PLOT_FONT_SIZE)
        ax.set_title(f'R2 scores\n{add_text}', fontsize=MAIN_PLOT_FONT_SIZE)
        ax.set_xticks(x_data)
        ax.set_xticklabels(labels)
        ax.tick_params(labelsize=BIG_LABELSIZE)
        ax.legend()
        if out_file_name is None:
            out_file_name = f'score_bars_{descr}'
        fig.savefig(f'{filepath[:filepath.rfind("/") + 1]}{out_file_name}.png', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def importance_bars(folder, name):
    # TODO: this function really depends on external effects, fix it
    importance_data = pd.read_excel(folder + 'importance_data.xlsx')
    flag = False
    for col in importance_data.columns:
        if '_' not in col:
            flag = True
            del importance_data[col]
    if flag:
        importance_data.to_excel(folder + 'importance_data.xlsx')
    imp_mass = ['const', 'simple', 'iterative', 'knn']
    columns = importance_data.columns.to_list()
    for i, imp in enumerate(imp_mass):
        assert columns[2 * i][columns[2 * i].rfind('_') + 1:] == imp
        assert columns[2 * i + 1][columns[2 * i + 1].rfind('_') + 1:] == imp, columns[i][columns[2 * i + 1].rfind('_') + 1:]
        fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
        ax.tick_params(labelsize=BIG_LABELSIZE)
        ax.barh(importance_data[columns[2 * i]], importance_data[columns[2 * i + 1]],
                color=color_arr_from_arr(np.array(importance_data[columns[2 * i + 1]]), init_color=(1.0, 1.0, 1.0), finish_color=(1.0, 0.9, 0.4), bottom=-0.25))
        # ax.barh(used_columns, model_name.feature_importances_, color='orange')
        # ax.tick_params(labelrotation=45)
        ax.set_title(f'Feature importance {name}', fontsize=MAIN_PLOT_FONT_SIZE)
        fig.savefig(f'{folder}feature_importance_{imp}_{name}.png', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def bars_for_descr_analysis(folder=''):
    graph_data = pd.read_excel(folder + 'descr_analysis.xlsx')
    # graph_data.to_excel('bar_check.xlsx')
    columns = graph_data['descriptor']
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=BIG_LABELSIZE)
    ax.tick_params(axis='y', labelsize=BIG_LABELSIZE)
    ax.bar(columns, graph_data['r2_arr'], color=color_arr_from_arr(np.array(graph_data['r2_arr'])))
    plt.savefig(folder + 'score_all_by_all.png', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=BIG_LABELSIZE)
    ax.tick_params(axis='y', labelsize=BIG_LABELSIZE)
    ax.bar(columns, graph_data['statistic'],
           color=color_arr_from_arr(np.array(graph_data['statistic']), init_color=(0.05, 0.05, 0.3),
                                    finish_color=(0.9, 0.8, 0.6), bottom=0., up=1.))
    fig.savefig(folder + 'statistic_.png', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def get_descr_distribution_picture(frame, descr, out_folder='', bins=50, color='#ffba54'):
    fig, ax = plt.subplots(figsize=MAIN_FIG_SIZE)
    data = np.array(frame.loc[dict_ind[descr], descr])
    total_values = data[data is not np.nan].size
    ax.hist(data, bins=bins, color=color, histtype='bar')
    ax.set_xlabel(f'Value of {descr}', fontsize=MAIN_PLOT_FONT_SIZE)
    ax.set_ylabel('Count', fontsize=MAIN_PLOT_FONT_SIZE)
    ax.set_title(f'{descr} distribution\ntotal number of values: {total_values}', fontsize=MAIN_PLOT_FONT_SIZE)
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.savefig(f'{out_folder}{descr}.png', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def get_descr_picture(df, columns, descr='', all_table=False, out_folder=''):
    if all_table:
        matr = df.T.isna().astype('int64').to_numpy()
        title = 'График разреженности\nбазы данных'
    else:
        assert len(descr), 'expect descriptor`s name, received nothing'
        matr = df.loc[dict_ind[descr], columns].T.isna().astype('int64').to_numpy()
        title = f'График, наглядно демонстрирующий разреженность матрицы данных составленной из столбцов в которых {descr} notNull'
    f, ax = plt.subplots(figsize=MAIN_FIG_SIZE)
    ax.tick_params(labelsize=BIG_LABELSIZE)
    ax = sns.heatmap(matr, linewidths=0.1, xticklabels=False, yticklabels=columns, ax=ax, cbar=False, cmap=['blue', 'white'])
    ax.set_title(title, fontsize=MAIN_PLOT_FONT_SIZE)
    if all_table:
        f.savefig(f'{out_folder}/all_table_picture.png', dpi=MAIN_DPI, bbox_inches='tight')
    else:
        f.savefig(f'{out_folder}/{descr}_picture.png', dpi=MAIN_DPI, bbox_inches='tight')
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
                fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
                ax.plot(data.loc[data.index[data['PaperID'] == p], used_descr], data.loc[data.index[data['PaperID'] == p], target_descr], '-o', c=np.random.choice(['red', 'green', 'blue', 'yellow', 'pink']))
                ax.set_xlabel(used_descr)
                ax.set_ylabel(target_descr)
                fig.savefig(f'{out_folder}{used_descr}_{target_descr}_paper{p}.png', dpi=MAIN_DPI, bbox_inches='tight')
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
        fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
        ax.scatter(data[used_descr], data[target_descr], c=color_data)
        ax.set_xlabel(used_descr)
        ax.set_ylabel(target_descr)
        fig.savefig(f'{out_folder}{used_descr}_{target_descr}_picture.png', dpi=MAIN_DPI, bbox_inches='tight')
        plt.close(fig)


def arts_descrs_picture(frame, out_folder='', hist_or_bar='hist'):
    _, counts = np.unique(frame['PaperID'], return_counts=True)
    fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
    if hist_or_bar == 'hist':
        _, frequency = np.unique(counts, return_counts=True)
        max_counts = np.max(counts)
        ax.hist(counts, bins=max_counts - 1)
        ax.set_xlabel('Number of experiments \nin article')
        ax.set_ylabel('Number of articles')
        ax.set_xticks(np.arange(max_counts + 1))
    elif hist_or_bar == 'bar':
        unique_counts, frequency = np.unique(counts, return_counts=True)
        ax.bar(unique_counts, frequency, color=(0.8, 0.8, 0.25))
        ax.tick_params(labelsize=BIG_LABELSIZE)
        ax.set_xlabel('Number of experiments in article', fontsize=16)
        ax.set_ylabel('Number of articles', fontsize=16)
    ax.set_yticks(np.arange(np.max(frequency) + 1))
    fig.savefig(f'{out_folder}arts_descrs_picture.png', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(fig)


def get_articles_picture(folder, score_mod='relative'):
    if score_mod == 'relative':
        vmin = -1
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
    f, ax = plt.subplots(figsize=MAIN_FIG_SIZE)
    ax = sns.heatmap(frame.to_numpy()[:, 2:].astype('float64'), linewidths=0.1, linecolor='black',
                     xticklabels=frame.columns[2:], yticklabels=list(columns['descr']),
                     ax=ax, cbar=True, vmax=vmax, vmin=vmin, cmap='summer_r')
    ax.tick_params(labelsize=BIG_LABELSIZE)
    f.tight_layout()
    f.savefig(folder + 'articles_picture_' + score_mod + '.png', dpi=MAIN_DPI, bbox_inches='tight')
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
                         fileName=f'{out_folder}{name}_scatter.png')


def quality_heatmap(folder=''):
    df = pd.read_excel(folder + 'qheatmap_data.xlsx')
    del df['Unnamed: 0']
    matrix_r2 = df.to_numpy()
    features = df.columns
    f, ax = plt.subplots(1, figsize=(MAIN_FIG_SIZE[0], 1.2 * MAIN_FIG_SIZE[1]))
    ax = sns.heatmap(matrix_r2, xticklabels=features, vmin=0., vmax=0.8, yticklabels=features)
    ax.tick_params(axis='x', labelrotation=90, labelsize=BIG_LABELSIZE)
    ax.tick_params(axis='y', labelsize=BIG_LABELSIZE)
    # ax.set_xlabel('predicted')
    # ax.set_ylabel('true')
    f.savefig(folder + 'heat_map.png', dpi=MAIN_DPI, bbox_inches='tight')
    plt.close(f)


def all_plots(main_path=''):
    # arts descrs picture
    frame = pd.read_excel(main_path + 'origin_frame.xlsx')
    frame.index = pd.read_excel(main_path + 'origin_frame.xlsx', usecols='A')['Unnamed: 0']
    del frame['Unnamed: 0']
    arts_descrs_picture(frame, hist_or_bar='bar', out_folder=main_path)
    current_path = main_path + 'descr_analysis/'
    bars_for_descr_analysis(folder=current_path)
    for fold in 'no_filter/', 'filter/':
        current_path = main_path + fold
        frame = pd.read_excel(current_path + 'x_file.xlsx')
        frame.index = pd.read_excel(current_path + 'x_file.xlsx', usecols='A')['Unnamed: 0']
        del frame['Unnamed: 0']
        columns = frame.columns
        # sparsity pictures
        count_sparsity(frame.loc[:, remove_many(columns, ['PaperID', 'Bad'])], create_bar=True, out_folder=current_path)
        get_descr_picture(frame, columns, all_table=True, out_folder=current_path)
        # article_analysis pictures
        for mod in 'relative', 'mean':
            get_articles_picture(current_path, score_mod=mod)
        # get_result pictures
        bar_descrs = ['H']
        bar_for_get_result(current_path, bar_descrs=clean_list(frame, bar_descrs, inplace=False))
    current_path = main_path + 'filter/'
    # importance pictures
    importance_bars(current_path, 'H')
    # quality heatmap
    quality_heatmap(folder=current_path)

# def get_all_plots(frame, out_folder=''):
#     # sparsity for descriptors plots
#     get_descr_picture(frame, frame.columns, all_table=True, out_folder=out_folder)
#     get_descr_picture(frame, exp_descr, 'H', out_folder=out_folder)
#     get_descr_picture(frame, exp_descr, 'E', out_folder=out_folder)
#     get_descr_picture(frame, exp_descr, 'CoatMu', out_folder=out_folder)
#     get_descr_picture(frame, exp_descr, 'CoatIntStress', out_folder=out_folder)
#     # sparsity per descriptors bar
#     count_sparsity(frame, create_bar=True)
#     # descriptors distribution
#     for name in frame.columns:
#         get_descr_distribution_picture(frame, name, folder_to_put_file=f'{out_folder}DescrsDistribution/')
#     # normalize
#     get_normal(frame)
#     clean_list(frame, str_descr)
#     use_label_encoder(frame, str_descr)
#     # scatter_plots
#     features = exp_descr + struct_descr + mech_descr
#     features.remove('H')
#     for pr in itertools.combinations(features, 2):
#         descriptor.plot_descriptors_2d(recovery_data(frame, frame.columns, recovery_method='const', fill_value=-2), pr,
#                                        ['H'], folder_prefix=f'{out_folder}H_folder', plot_only="data and quality")
#     # articles picture
#     score_mod = 'mean'
#     article_analysis(frame, articles_names, remove_many(list(frame.columns), ['PaperID', 'Bad', 'CheckSum_N2Press_SubT',
#                                                                               'CheckSum_DeposRate_VoltBias']),
#                      out_file_name=f'{out_folder}ArticleAnalysis.xlsx', predict_to_file=False, model_name='ExtraTR',
#                      score_mod=score_mod)
#     get_articles_picture(f'{out_folder}ArticleAnalysis.xlsx', f'{out_folder}articles_picture.png', score_mod=score_mod)
