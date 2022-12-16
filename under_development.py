# import sklearn as skl

import pyfitit

# from TiN_utils import *
from TiN_frame_process import *
# from TiN_Dataset import TiN_Dataset


def get_frame_of_ratios(frame, articles, num_rows_per_art=10, mode='ratio'):
    frame.to_excel('x_file_1.xlsx')
    out = pd.DataFrame(np.full(frame.shape, np.nan))
    out.columns = frame.columns
    recove_frame = recover_dataframe(frame, frame.columns, recovery_method='const', fill_value=1)
    ind_row = 0
    cols = lists_difference(recove_frame.columns, nominal_descr + ['PaperID'])
    nominal_cols = del_from_list_if_not_in_df(recove_frame, nominal_descr)
    if mode == 'ratio':
        for name in articles:
            art = recove_frame.loc[recove_frame['PaperID'] == name, cols]
            nominal_part = recove_frame.loc[recove_frame['PaperID'] == name, nominal_cols]
            num_exp = art.shape[0]
            if num_exp > 1:
                art = art.to_numpy()
                nominal_part = nominal_part.to_numpy()
                allinds = pyfitit.utils.comb_index(num_exp, 2, repetition=False)
                np.random.shuffle(allinds)
                allinds = allinds[:3 * (num_rows_per_art // 2), :]
                i = 0
                while i < len(allinds):
                    inds = allinds[i]
                    try:
                        ratios = art[inds[0]] / art[inds[1]]
                        out.loc[ind_row, cols] = ratios
                        out.loc[ind_row, nominal_cols] = (nominal_part[inds[0]] == nominal_part[inds[1]]).astype('int32')
                        out.loc[ind_row, 'PaperID'] = f'{name} {inds[0]}/{inds[1]}'
                        ind_row += 1
                    except ZeroDivisionError:
                        i -= 1
                    try:
                        ratios = art[inds[1]] / art[inds[0]]
                        out.loc[ind_row, cols] = ratios
                        out.loc[ind_row, nominal_cols] = (nominal_part[inds[0]] == nominal_part[inds[1]]).astype('int32')
                        out.loc[ind_row, 'PaperID'] = f'{name} {inds[1]}/{inds[0]}'
                        ind_row += 1
                    except ZeroDivisionError:
                        i -= 1
                    i += 3
    elif mode == 'difference':
        for name in articles:
            art = recove_frame.loc[recove_frame['PaperID'] == name, cols]
            nominal_part = recove_frame.loc[recove_frame['PaperID'] == name, nominal_cols]
            num_exp = art.shape[0]
            if num_exp > 1:
                art = art.to_numpy()
                nominal_part = nominal_part.to_numpy()
                allinds = pyfitit.utils.comb_index(num_exp, 2, repetition=False)
                np.random.shuffle(allinds)
                allinds = allinds[:num_rows_per_art, :]
                for i in range(len(allinds)):
                    inds = allinds[i]
                    ratios = art[inds[0]] - art[inds[1]]
                    out.loc[ind_row, cols] = ratios
                    out.loc[ind_row, nominal_cols] = (nominal_part[inds[0]] != nominal_part[inds[1]]).astype('int32')
                    out.loc[ind_row, 'PaperID'] = f'{name} {inds[0]}-{inds[1]}'
                    ind_row += 1
    else:
        assert False, f'invalid value for mode: {mode}'
    if mode == 'ratio':
        out[out == 1] = np.nan
    elif mode == 'difference':
        out[out == 0] = np.nan
    out.to_excel('x_file.xlsx')
    return out


# def data_analysis(frame):
#     print("You should preset flags for data_analysis")
#     print("Enter 1 if you want to set flag \"True\"")
#     print("Or press Enter if you want to set flag \"False\"")
#     flag_default = bool(input("use default flags (enter 1 to use default else press Enter): "))
#     flag_check = flag_default or bool(input("perform check (enter 1 to execute else press Enter): "))
#     # flag_filter = bool(input("perform filtration (enter 1 to execute else press Enter): "))
#     # flag_descr_picture = bool(input("create_descr picture (enter 1 to execute else press Enter): "))
#     # flag_sparsity = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_OneHotEncoder = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_LabelEncoder = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_2dplos = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_quality = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_descr_analysis = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_article_analysis = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     # flag_get_result = bool(input("building 2d-plots (enter 1 to execute else press Enter): "))
#     if flag_check:
#         frame = generate_df_with_diff_dependencies(frame, main_chance=0.2)
#     pass


if __name__ == '__main__':
    # USAGE OF UNFINISHED FUNCTIONS
    # score_mode = 'relative'
    # score_mod = 'relative'
    # article_analysis(x, articles_names, lists_difference(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), out_file_name='Check_analysis.xlsx', predict_file_name='Useless.xlsx', predict_to_file=False, model_name='RidgeCV', score_mod=score_mod)
    # article_analysis_table('Check_analysis.xlsx', 'check_picture.png', score_mod=score_mod)
    # x.to_excel('Check_x.xlsx')

    # arts_descrs_picture(x, hist_or_bar='bar', out_folder=f'{PLOT_FOLDER}/')  # figS1

    # article_analysis(x, articles_names, lists_difference(list(x.columns), ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), predict_to_file=False, model_name='ExtraTR', score_mod=score_mode,
    #       out_folder='22_04_results')
    # article_analysis_table('22_04_results/', score_mod=score_mode)

    # current_cols = lists_difference(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias'])
    # descr_analysis(x, current_cols)
    # descr_analysis_stat_bars()

    # data_analysis(x)

    # OTHER UNUSED CODE

    # RECOVERING
    # recove_x = recover_dataframe(x, lists_difference(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), lists_difference(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), recovery_method='iterative', num_iter=500)
    # recove_x.to_excel('ResultRecovering.xlsx')

    # x = recover_dataframe(x, exp_descr, recovery_method='const', fill_value=FILL_VALUE)

    # x['SubType'][x['SubType'].isna()] = -1
    # meanH_for_subType = x.groupby('SubType')['H'].mean()
    # rH = [x.loc[i, 'H']/meanH_for_subType[x.loc[i, 'SubType']] for i in range(x.shape[0])]
    # x['rH'] = rH
    # mech_descr.append('rH')
    # dict_ind['rH'] = copy.deepcopy(dict_ind['H'])

    # x.loc[:, del_from_list_if_not_in_df(x, nominal_descr, inplace=False)].to_excel('x_file_3.xlsx')

    # mode = 'ratio'
    # flag_encoder = False
    # FILL_VALUE = 1
    # if mode == 'difference':
    #     FILL_VALUE = 0
    # x = get_frame_of_ratios(x, articles_names, mode=mode)

    # find_large_intersect_nominal_descr(x, del_from_list_if_not_in_df(x, nominal_descr, inplace=False))
    # find_large_intersect_nominal_descr(x, ['SubType'])
    # new_x = x.loc[x.index[x['React_or_not'] == 1]]
    # new_x = new_x.loc[new_x.index[new_x['Balanc_or_not'] == 0]]
    # new_x = new_x.loc[new_x.index[new_x['SubType'] == 'steel']]
    # new_x = new_x.loc[new_x.index[new_x['SubLayer'] == 0]]
    # -----
    # new_x = x.loc[x.index[x['React_or_not'] == 1]]
    # new_x = new_x.loc[new_x.index[new_x['Balanc_or_not'] == 1]]
    # new_x = new_x.loc[new_x.index[new_x['SubType'] == 'Si']]
    # new_x = new_x.loc[new_x.index[new_x['SubLayer'] == 1]]
    # -----
    # new_x = x.loc[x.index[x['SubType'] == 'steel']]
    # new_x = x.loc[x.index[x['SubType'] == 'Si']]
    # -----
    # x = new_x

    # for d in exp_descr:
    #     descr_correlate_picture(x, d, 'H', out_folder='Correlations/')

    # degree of main_chance:
    # high: 0.5
    # medium: 0.3
    # small: 0.2
    # very_small: 0.08
    # x = generate_df_with_diff_dependencies(x, main_chance=0.2)

    # %INVERSE PROBLEM
    # fit_many_imputers_and_models(x, lists_difference(exp_descr, nominal_descr), ['H'], fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=False, folder='InverseProblem_12_01_22/', articles=articles_names, true_vs_predicted_picture_fname=False)
    # fit_one_target_others_features(x, lists_difference(exp_descr, nominal_descr), ['H'], fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=False, folder='InverseProblem_12_01_22/', articles=articles_names, true_vs_predicted_picture_fname=False)
    # fit_many_imputers_and_models(x, mech_descr, exp_descr, fill_value=FILL_VALUE, crossval_typo='3:1', folder='InverseProblem_12_01_22/', articles=articles_names, true_vs_predicted_picture_fname=False)

    # descriptor.getAnalyticFormulasForGivenFeatures(recover_dataframe(x, exp_descr, recovery_method='const', fill_value=FILL_VALUE).loc[dict_ind['H'], :], exp_descr, 'H', output_file='formulas.txt')

    # MANY of global lists, dicts are currently in the file 'TiN_frame_process.py'

    # DATA READING
    # filepath = './DataTable_Guda_2.xlsx'
    # filepath = './DataTable_Guda_3.xlsx'

    # CHANGE parameter USECOLS if number of experiments has changed
    # df_TiN_coatings = pd.read_excel('./DataTable_Guda_2.xlsx', usecols=f'F:EN', skiprows=1)  # 139 samples
    # df_TiN_coatings = pd.read_excel('./DataTable_Guda_3.xlsx', usecols=f'F:JZ', skiprows=1)  # 281 samples

    # FILTERING
    # descr_names = pd.read_excel(filepath, usecols='C', skiprows=1).to_numpy().reshape(1, -1)[0]

    # arrays of indexes of good samples
    # 0.1: soft filter; 2.1: hard filter
    # values in article: -2. for fig1, S1, S2; 2.1 for fig2; 0.1, 2.1 for fig3, but mostly used prepared data;
    # 2.1 for fig 4, 5, but mostly used prepared data;
    # desr_rating = pd.read_excel(filepath, usecols='E', skiprows=1).to_numpy()
    # filter_rubbish = (desr_rating > -0.9).reshape(1, -1)[0]
    # good = (desr_rating > 2.1).reshape(1, -1)[0]
    # good_names = descr_names[good]
    #
    # df_TiN_coatings = df_TiN_coatings.T
    # df_TiN_coatings.reset_index(drop=True, inplace=True)
    # df_TiN_coatings.columns = descr_names
    # good_exp_inds = np.arange(df_TiN_coatings.shape[0])[df_TiN_coatings['Bad'].isna()]

    # if not os.path.exists(PLOT_FOLDER):
    #     os.makedirs(PLOT_FOLDER)
    # all_plots(df_TiN_coatings,
    #           PLOT_FOLDER,
    #           filter_ops=[True, True, good_names, good_exp_inds],
    #           # results_ops={'data_file_path': '22_04_results/unfiltered/ModelingResults.xlsx',
    #           #              'one_more_file_path': '22_04_results/filtered/ModelingResults.xlsx',
    #           #              'out_file_name': 'fig3',
    #           #              'bar_descrs': del_from_list_if_not_in_df(x, ['H'], inplace=False),
    #           #              'add_text_plot': [(0.42, 0.95, 'ExtraTrees'), (0.68, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
    #           #              'text_plot_ops': {'transform': True, 'fontweight': 'bold'}},
    #           # importance_ops={'in_path': '22_04_results/filtered/importance_data.xlsx', 'name': 'H'},
    #           qheatmap_ops=('.', ))

    # good_names = del_from_list_if_not_in_df(df_TiN_coatings, good_names, False)

    # df_TiN_coatings = filter_df(df_TiN_coatings, delete_names=True, delete_exps=True, good_names=good_names, good_exps=good_exp_inds)
    # df_TiN_coatings.reset_index(drop=True, inplace=True)
    # delete_empty_invalid_descriptors(df_TiN_coatings, descr_names)
    #
    # # STATISTIC, INFORMATION EXTRACTION
    # arts, inds = np.unique(df_TiN_coatings['PaperID'], return_index=True)
    # inds = df_TiN_coatings.index[inds]
    # inds = np.sort(inds)
    # articles_names = df_TiN_coatings.loc[inds, 'PaperID']
    # print(articles_names.to_numpy())
    #
    # count_sparsity_plotting_bar(df_TiN_coatings.loc[:, lists_difference(df_TiN_coatings.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
    #                             out_file_path=f'{PLOT_FOLDER}/fig2{EXT}')

    # descr_sparcity_table(x, x.columns, all_table=True, out_folder=PLOT_FOLDER)    # fig1

    # x.to_excel('Check0.xlsx')

    # for name in x.columns:
    #     get_descr_distribution_picture(x, name, out_folder='22_04_results/descrs_distribution/')

    # ENCODE STRINGS, NORMALIZE
    # str_descr = del_from_list_if_not_in_df(df_TiN_coatings, str_descr)
    # if USE_ENCODER:
    #     apply_label_encoder(df_TiN_coatings, str_descr)

    # normalize_frame(df_TiN_coatings, norm_nominal=True)

    # qheatmap_data(df_TiN_coatings, target_name='H', feature_names=np.array(best_features),
    #               dest_folder='221207_data/',
    #               model_typo='ExtraTR', recovering='const')
    # quality_heatmap(out_folder=PLOT_FOLDER)    # fig5

    # get_scatter_plots(recover_dataframe(x, 'H', recovery_method='const', fill_value=FILL_VALUE), 'H', ['SubType', 'ChambPress', 'CathDist', 'ResidPress'], out_folder='Scatters/')

    # try_sets_of_1_2_3_4_descrs(x, exp_descr, 'H', model_name='ExtraTR', flags=(1, 1, 1, 1, 0), cv_parts=4, out_folder='NoFilterTry_22_04_14/', recovery_method='const', fill_value=FILL_VALUE)

    # fit_many_imputers_and_models(df_TiN_coatings, ['H'], del_from_list_if_not_in_df(df_TiN_coatings, exp_descr), fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=True, folder='221209/filtered', articles=articles_names, true_vs_predicted_picture_fname=f'true_vs_predicted')

    # crosval_fit_for_each_target(recover_dataframe(x, exp_descr, recovery_method='knn', fill_value=FILL_VALUE), mech_descr, exp_descr, crossval_typo='3:1', mod='all', true_vs_predicted_picture_fname='Scatters/Scatters_predict/table3/')

    # bar_descrs = ['H', 'E', 'CoatMu', 'CritLoad']
    # bar_descrs = ['H']
    # add_text={'s': 'filtered frame', 'x': 0.0, 'y': 0.9}
    # training_results_bar('22_04_results/unfiltered/ModelingResults.xlsx',  # fig 3
    #                      out_folder=PLOT_FOLDER,
    #                      one_more_file_path='22_04_results/filtered/ModelingResults.xlsx',
    #                      out_file_name='fig3',
    #                      bar_descrs=del_from_list_if_not_in_df(x, bar_descrs, inplace=False),
    #                      add_text_plot=[(0.56, 0.95, 'ExtraTrees'), (0.72, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
    #                      text_plot_ops={'transform': True},
    #                      )
    # importance_bars('22_04_results/unfiltered/importance_data.xlsx', '221010_pictures', 'H')   # fig4

    pass

