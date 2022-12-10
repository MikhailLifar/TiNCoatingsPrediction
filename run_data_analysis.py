# import itertools
# import sys
# import os

# import sklearn as skl
# from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import RidgeCV, LogisticRegression
# from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.svm import SVC, SVR
# from fancyimpute import Solver, NuclearNormMinimization, MatrixFactorization, IterativeSVD, SimpleFill, SoftImpute, BiScaler, KNN, SimilarityWeightedAveraging
# from datawig import SimpleImputer as DwImputer

from pyfitit import *

from TiN_frame_process import *
from TiN_plots import *


def get_regressor_classifier(model_typo: str):
    if model_typo == 'RidgeCV':
        model_regr = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
        model_classifier = LogisticRegression()
    elif model_typo == 'ExtraTR':
        model_regr = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_typo == 'Gauss':
        kernel = C(1.0) * RBF(1.0)
        model_regr = GaussianProcessRegressor(kernel=kernel)
        model_classifier = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=0)
    elif model_typo == 'SVM':
        model_regr = SVR(kernel='rbf', C=10, epsilon=0.1)
        model_classifier = SVC(kernel='rbf', C=10)
    else:
        assert False, f'invalid value for model_typo: {model_typo}'
    return model_regr, model_classifier


def crosval_fit_for_each_target(df, target_names, feature_names, model_regr, model_classifier,
                                folder_to_save_results='./', results_fname='ModelingResults.xlsx',
                                which_target_values_use: str = 'original', crossval_typo: str = 'LOO arts',
                                return_r2=True, save_excel=False, true_vs_predicted_picture_fname: str = '',
                                count_importance_to_file=False,
                                **kwargs):
    """
    The function runs fitting for each name in target_names,
    taking feature names as inputs for fitting.
    Optionally the function saves TrueVsPredicted file (.xlsx),
    draw TrueVsPredicted picture
    and saves feature_importance (in .xlsx)

    :param df:
    :param target_names:
    :param feature_names:
    :param model_regr:
    :param model_classifier:
    :param folder_to_save_results:
    :param results_fname:
    :param which_target_values_use:
    :param crossval_typo:
    :param return_r2:
    :param save_excel:
    :param true_vs_predicted_picture_fname:
    :param count_importance_to_file:
    :param kwargs:
    :return: r2-scores
    """

    # preparations
    if not isinstance(target_names, list):
        target_names = list(target_names)
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    samples_num = df.shape[0]
    dict_r2 = dict()
    if save_excel:
        output = pd.DataFrame()

    for name in target_names:

        # preparations: choose the model (regressor or classifier, depending on target)
        if name in str_descr:
            model_for_the_name = model_classifier
        else:
            model_for_the_name = model_regr
        if which_target_values_use == 'original':
            idxs = dict_ind[name]
        elif which_target_values_use == 'all':
            idxs = list(range(samples_num))
        else:
            assert False, f'invalid value for which_target_values_use: {which_target_values_use}'
        true_targets = np.array([None] * samples_num)
        predicted = np.array([None] * samples_num)
        true_targets[idxs] = df.loc[idxs, name]
        score_can_be_calculated = True

        # cross-validation: fit and prediction
        if crossval_typo == 'LOO exps':  # Leave One Out, per samples
            for i, idx in enumerate(idxs):
                model_for_the_name.fit(df.loc[del_per_index(idxs, i), feature_names], df.loc[del_per_index(idxs, i), name])
                arr = model_for_the_name.predict(df.loc[idxs[i], feature_names].to_numpy().reshape(1, -1))
                predicted[idx] = arr[0]
                # if model_str == 'RidgeCV':
                #     print(model.alpha_)
        elif crossval_typo == crossval_typo[0] + ':1':
            assert crossval_typo[0] in '3456789', f'invalid value for crossval_typo: {crossval_typo}'
            cv = int(crossval_typo[0])
            if len(idxs) < (cv + 1) * 5:
                print(f'Warning! Number of experiments is too small for {name}')
                predicted[idxs] = np.mean(true_targets[idxs]) - 1
            else:
                idxs = np.array(idxs)
                np.random.shuffle(idxs)
                previous_divider = 0
                divider = idxs.size // (cv + 1) + 1
                for i in range(cv + 1):
                    test = idxs[previous_divider:divider]
                    train = np.array(remove_many(idxs, test))
                    model_for_the_name.fit(df.loc[train, feature_names], df.loc[train, name])
                    predicted[test] = model_for_the_name.predict(df.loc[test, feature_names])
                    previous_divider = divider
                    divider += idxs.size // (cv + 1) + 1
                    divider = min(idxs.size, divider)
        elif crossval_typo == 'LOO arts':    # Leave One Out, per articles
            for name_art in kwargs['articles']:
                art = df.loc[df['PaperID'] == name_art]
                ind = np.where(df['PaperID'] != name_art)[0]
                descr_inds_art = clean_list_of_names(art.T, idxs, inplace=False)
                ind_notnull = np.intersect1d(idxs, ind)
                if (ind_notnull.size > 0) and (len(descr_inds_art) > 0):
                    model_for_the_name.fit(df.loc[ind_notnull, feature_names], df.loc[ind_notnull, name])
                    predicted[descr_inds_art] = model_for_the_name.predict(art.loc[descr_inds_art, feature_names])
                elif ind_notnull.size == 0:
                    score_can_be_calculated = False
        else:
            assert False, f'invalid value for crossval_typo: {crossval_typo}'

        if save_excel:
            if name in str_descr:
                output[f'{name}_true'] = true_targets
                output[f'{name}_predicted'] = predicted
            else:
                true_targets[idxs] = true_targets[idxs] * dict_norm[name][1] + dict_norm[name][0]
                output[f'{name}_true'] = true_targets
                predicted[idxs] = predicted[idxs] * dict_norm[name][1] + dict_norm[name][0]
                output[f'{name}_predicted'] = predicted
                abs_error = copy.deepcopy(predicted)
                abs_error[idxs] = abs(predicted[idxs] - true_targets[idxs])
                output[f'{name}_absolute_error'] = abs_error

        if score_can_be_calculated:
            dict_r2[name] = scoreFast(true_targets[idxs], predicted[idxs])
        else:
            dict_r2[name] = np.nan

        if true_vs_predicted_picture_fname:
            if not save_excel:
                if name not in str_descr:
                    true_targets[idxs] = true_targets[idxs] * dict_norm[name][1] + dict_norm[name][0]
                    predicted[idxs] = predicted[idxs] * dict_norm[name][1] + dict_norm[name][0]
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.scatter(predicted, true_targets, c='red')
            ax.plot([-10, 100], [-10, 100], c='red')
            ax.set_xlabel(f'{name}_predicted')
            ax.set_ylabel(f'{name}_original')
            fig.savefig(f'{folder_to_save_results}/{true_vs_predicted_picture_fname}_{name}{EXT}')
            plt.close(fig)

        if count_importance_to_file and hasattr(model_for_the_name, 'feature_importances_'):
            importance_data = np.empty(len(feature_names), dtype=[('feature_name', 'U20'), ('importance', float)])
            for i in range(len(feature_names)):
                importance_data[i] = feature_names[i], model_for_the_name.feature_importances_[i]
            importance_data.sort(order='importance')
            # df = pd.read_excel(f'{folder_to_save_results}/importance_data.xlsx', index_col='???')
            df = pd.DataFrame()
            df['feature_' + kwargs["imp_name"]] = importance_data['feature_name']
            df['importance_' + kwargs["imp_name"]] = importance_data['importance']
            df.to_excel(f'{folder_to_save_results}/importance_data.xlsx', index_label='???')

    if save_excel:
        output.loc[-1, :] = np.nan
        for name in target_names:
            output.loc[-1, name + '_predict'] = dict_r2[name]
        output.sort_index(inplace=True)
        output.to_excel(f'{folder_to_save_results}/{results_fname}', index=False)
        return f'{folder_to_save_results}/{results_fname}'
    if return_r2:
        return dict_r2


def fit_many_imputers_and_models(df, target_names, feature_names, folder='', filename='ModelingResultsTable.xlsx',
                                 imputers=('const', 'simple', 'iterative', 'knn'),
                                 models=('ExtraTR', 'SVM', 'RidgeCV'),
                                 **keyargs):

    """
    Function receives imputers and models tuples,
    then for each pair of direct product of the tuples
    it runs crosval_fit_for_each_target

    :return: None
    """

    # imputers = ('simple', 'iterative', 'knn', 'soft_imp', 'const', 'matr_factoriz', 'similarity')
    # models = ('ExtraTR', 'RidgeCV', 'Gauss', 'SVM')
    # models = ('ExtraTR')
    res = pd.DataFrame()
    frame_imps_mass = []
    for name in imputers:
        frame_imps_mass += [name] * len(models)
    res['imputer'] = np.array(frame_imps_mass)
    res['model'] = np.array(models * len(imputers))
    dict_res = dict()
    for name in target_names:
        dict_res[name] = np.array([None] * (len(imputers) * len(models)))
    j = 0

    fill_value = keyargs['fill_value']
    del keyargs['fill_value']
    true_vs_pred_fname2 = true_vs_pred_fname = ''
    if 'true_vs_predicted_picture_fname' in keyargs:
        true_vs_pred_fname = keyargs['true_vs_predicted_picture_fname']
        del keyargs['true_vs_predicted_picture_fname']
    for imp in imputers:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recovery_data(df, feature_names, recovery_method=imp, fill_value=fill_value)
        for model_typo in models:
            model_regr, model_classifier = get_regressor_classifier(model_typo)
            if true_vs_pred_fname:
                true_vs_pred_fname2 = f'{true_vs_pred_fname}_{imp}_{model_typo}'
            dict_res_model = crosval_fit_for_each_target(recove_frame, target_names, feature_names, model_regr, model_classifier,
                                                         imp_name=imp, folder_to_save_results=folder,
                                                         true_vs_predicted_picture_fname=true_vs_pred_fname2,
                                                         **keyargs, )
            for name in dict_res_model:
                dict_res[name][j] = dict_res_model[name]
            j += 1
            print('Fitting completed')
    for name in target_names:
        res[name] = dict_res[name]
    res.T.to_excel(f'{folder}/{filename}')


def fit_one_target_others_features(dataframe, target_names, feature_names, folder='', filename='ModelingResultsTable.xlsx',
                                   imputers=('const', 'simple', 'iterative', 'knn'),
                                   models=('ExtraTR', 'RidgeCV', 'SVM'),
                                   **keyargs):

    """

    Each name from target names is predicted using
    feature_names and other target names

    :return: None
    """

    # imputers = ['simple', 'iterative', 'knn', 'soft_imp', 'const', 'matr_factoriz', 'similarity']
    # models = ['ExtraTR', 'RidgeCV', 'Gauss', 'SVM']
    res = pd.DataFrame()
    frame_imps_mass = []
    for name in imputers:
        frame_imps_mass += [name] * len(models)
    res['imputer'] = np.array(frame_imps_mass)
    res['model'] = np.array(models * len(imputers))
    dict_res = dict()
    for name in target_names:
        dict_res[name] = np.array([None] * (len(imputers) * len(models)))
    j = 0
    for imp in imputers:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recovery_data(dataframe, feature_names + target_names, recovery_method=imp, fill_value=keyargs['fill_value'])
        del keyargs['fill_value']
        for model_typo in models:
            model_regr, model_classifier = get_regressor_classifier(model_typo)
            for i, name in enumerate(target_names):
                dict_res_model = crosval_fit_for_each_target(
                    recove_frame, [name], feature_names + target_names[:i] + target_names[i + 1:],
                    model_regr, model_classifier,
                    imp_name=imp, **keyargs)
                if name in dict_res_model:
                    dict_res[name][j] = dict_res_model[name]
            j += 1
            print('Fitting completed')
    for name in target_names:
        res[name] = dict_res[name]
    res.T.to_excel(f'{folder}/{filename}')


def try_sets_of_1_2_3_4_descrs(frame, feature_names, target_name, model_typo, sets_flags=(1, 1, 1, 0, 0), cv_parts=5, dest_folder='', **kwargs):
    assert isinstance(target_name, str), 'parameter target_name should be string'
    assert kwargs['recovery_method'] != 'iterative', 'Only not iterative for this function'
    model_regr, model_class = get_regressor_classifier(model_typo)
    features = feature_names
    try:
        features.remove(target_name)
    except ValueError:
        print(f'{target_name} not in cols')
    for i in range(1, len(sets_flags) + 1):
        if sets_flags[i - 1]:
            descriptor.descriptor_quality(recovery_data(frame.loc[dict_ind[target_name], :],
                                                        feature_names, recovery_method=kwargs['recovery_method'], fill_value=kwargs['fill_value']),
                                          [target_name], features, model_regr=model_regr, model_class=model_class,
                                          feature_subset_size=i, cv_repeat=10, cv_parts_count=cv_parts,
                                          folder=f'{dest_folder}quality_{i}', shuffle=True)
            print(f'Combinations of {i} descriptors were fitted successfully')


def qheatmap_data(df, dest_folder, target_name, feature_names, model_typo='ExtraTR', recovering='const'):
    recove_frame = recovery_data(df, feature_names, recovery_method=recovering, fill_value=FILL_VALUE)
    # true_values = df.loc[dict_ind[descr], descr]
    # v = np.mean((true_values - np.mean(true_values)) ** 2)
    matrix_r2 = np.zeros((feature_names.size, feature_names.size))
    model_regr, model_classifier = get_regressor_classifier(model_typo)
    for i in range(feature_names.size):
        for j in range(i, feature_names.size):
            f1 = feature_names[i]
            f2 = feature_names[j]
            if f1 == f2:
                dict_value = crosval_fit_for_each_target(recove_frame, [target_name], [f1], model_regr, model_classifier,
                                                         crossval_typo='3:1', imp_name='')
            else:
                dict_value = crosval_fit_for_each_target(recove_frame, [target_name], [f1, f2], model_regr, model_classifier,
                                                         crossval_typo='3:1', imp_name='')
            matrix_r2[i][j] = dict_value[target_name]
            if f1 != f2:
                matrix_r2[j][i] = dict_value[target_name]
    out_df = pd.DataFrame()
    for ind, name in enumerate(feature_names):
        out_df[name] = matrix_r2[ind]
    out_df.to_excel(f'{dest_folder}/qheatmap_data.xlsx')


PLOT_FOLDER = f'./221207'
FILL_VALUE = -2
USE_ENCODER = True

if __name__ == '__main__':
    # MANY of global lists, dicts are currently in the file 'TiN_frame_process.py'

    # filepath = './DataTable_Guda_2.xlsx'
    filepath = './DataTable_Guda_3.xlsx'

    # CHANGE parameter USECOLS if number of experiments has changed
    # df_TiN_coatings = pd.read_excel('./DataTable_Guda_2.xlsx', usecols=f'F:EN', skiprows=1)  # 139 samples
    df_TiN_coatings = pd.read_excel('./DataTable_Guda_3.xlsx', usecols=f'F:JZ', skiprows=1)  # 281 samples

    descr_names = pd.read_excel(filepath, usecols='C', skiprows=1).to_numpy().reshape(1, -1)[0]

    # arrays of indexes of good samples
    # 0.1: soft filter; 2.1: hard filter
    # values in article: -2. for fig1, S1, S2; 2.1 for fig2; 0.1, 2.1 for fig3, but mostly used prepared data;
    # 2.1 for fig 4, 5, but mostly used prepared data;
    desr_rating = pd.read_excel(filepath, usecols='E', skiprows=1).to_numpy()
    filter_rubbish = (desr_rating > -0.9).reshape(1, -1)[0]
    good = (desr_rating > 2.1).reshape(1, -1)[0]
    good_names = descr_names[good]

    df_TiN_coatings = df_TiN_coatings.T
    df_TiN_coatings.reset_index(drop=True, inplace=True)
    df_TiN_coatings.columns = descr_names
    good_exp_inds = np.arange(df_TiN_coatings.shape[0])[df_TiN_coatings['Bad'].isna()]

    # if not os.path.exists(PLOT_FOLDER):
    #     os.makedirs(PLOT_FOLDER)
    # all_plots(df_TiN_coatings,
    #           PLOT_FOLDER,
    #           filter_ops=[True, True, good_names, good_exp_inds],
    #           # results_ops={'data_file_path': '22_04_results/unfiltered/ModelingResults.xlsx',
    #           #              'one_more_file_path': '22_04_results/filtered/ModelingResults.xlsx',
    #           #              'out_file_name': 'fig3',
    #           #              'bar_descrs': clean_list_of_names(x, ['H'], inplace=False),
    #           #              'add_text_plot': [(0.42, 0.95, 'ExtraTrees'), (0.68, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
    #           #              'text_plot_ops': {'transform': True, 'fontweight': 'bold'}},
    #           # importance_ops={'in_path': '22_04_results/filtered/importance_data.xlsx', 'name': 'H'},
    #           qheatmap_ops=('.', ))

    # good_names = clean_list_of_names(df_TiN_coatings, good_names, False)

    df_TiN_coatings = filter_df(df_TiN_coatings, delete_names=True, delete_exps=True, good_names=good_names, good_exps=good_exp_inds)
    df_TiN_coatings.reset_index(drop=True, inplace=True)
    delete_empty_invalid_descriptors(df_TiN_coatings, descr_names)

    arts, inds = np.unique(df_TiN_coatings['PaperID'], return_index=True)
    inds = df_TiN_coatings.index[inds]
    inds = np.sort(inds)
    articles_names = df_TiN_coatings.loc[inds, 'PaperID']
    print(articles_names.to_numpy())

    count_sparsity_plotting_bar(df_TiN_coatings.loc[:, remove_many(df_TiN_coatings.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                                out_file_path=f'{PLOT_FOLDER}/fig2{EXT}')

    # descr_sparcity_table(x, x.columns, all_table=True, out_folder=PLOT_FOLDER)    # fig1

    # x.to_excel('Check0.xlsx')

    # for name in x.columns:
    #     get_descr_distribution_picture(x, name, out_folder='22_04_results/descrs_distribution/')

    clean_list_of_names(df_TiN_coatings, str_descr)
    if USE_ENCODER:
        apply_label_encoder(df_TiN_coatings, str_descr)

    normalize_frame(df_TiN_coatings, norm_nominal=True)

    # print(dict_labels)

    # qheatmap_data(df_TiN_coatings, target_name='H', feature_names=np.array(best_features),
    #               dest_folder='221207_data/',
    #               model_typo='ExtraTR', recovering='const')
    # quality_heatmap(out_folder=PLOT_FOLDER)    # fig5

    # get_scatter_plots(recovery_data(x, 'H', recovery_method='const', fill_value=FILL_VALUE), 'H', ['SubType', 'ChambPress', 'CathDist', 'ResidPress'], out_folder='Scatters/')

    # try_sets_of_1_2_3_4_descrs(x, exp_descr, 'H', model_name='ExtraTR', flags=(1, 1, 1, 1, 0), cv_parts=4, out_folder='NoFilterTry_22_04_14/', recovery_method='const', fill_value=FILL_VALUE)

    fit_many_imputers_and_models(df_TiN_coatings, ['H'], clean_list_of_names(df_TiN_coatings, exp_descr, inplace=False), fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=True, folder='221209/filtered', articles=articles_names, true_vs_predicted_picture_fname=f'true_vs_predicted')

    # bar_descrs = ['H', 'E', 'CoatMu', 'CritLoad']
    bar_descrs = ['H']
    # add_text={'s': 'filtered frame', 'x': 0.0, 'y': 0.9}
    # training_results_bar('22_04_results/unfiltered/ModelingResults.xlsx',  # fig 3
    #                      out_folder=PLOT_FOLDER,
    #                      one_more_file_path='22_04_results/filtered/ModelingResults.xlsx',
    #                      out_file_name='fig3',
    #                      bar_descrs=clean_list_of_names(x, bar_descrs, inplace=False),
    #                      add_text_plot=[(0.56, 0.95, 'ExtraTrees'), (0.72, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
    #                      text_plot_ops={'transform': True},
    #                      )
    # importance_bars('22_04_results/unfiltered/importance_data.xlsx', '221010_pictures', 'H')   # fig4

    # crosval_fit_for_each_target(recovery_data(x, exp_descr, recovery_method='knn', fill_value=FILL_VALUE), mech_descr, exp_descr, crossval_typo='3:1', mod='all', true_vs_predicted_picture_fname='Scatters/Scatters_predict/table3/')

    # descriptor.getAnalyticFormulasForGivenFeatures(recovery_data(x, exp_descr, recovery_method='const', fill_value=FILL_VALUE).loc[dict_ind['H'], :], exp_descr, 'H', output_file='formulas.txt')
