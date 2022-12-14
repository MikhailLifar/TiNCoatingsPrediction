# import itertools
# import sys
# import os

import sklearn as skl
# from sklearn.experimental import enable_iterative_imputer

from sklearn.linear_model import RidgeCV, LogisticRegression
# from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.svm import SVC, SVR
# from fancyimpute import Solver, NuclearNormMinimization, MatrixFactorization, IterativeSVD, SimpleFill, SoftImpute, BiScaler, KNN, SimilarityWeightedAveraging

from TiN_frame_process import *
from TiN_plots import *
from TiN_Dataset import TiN_Dataset


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


def crosval_fit_for_each_target(df, dict_idxs, dict_norm, target_names, feature_names, model_regr, model_classifier,
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

    :param dict_idxs:
    :param dict_norm:
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
            idxs = dict_idxs[name]
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
                    train = np.array(lists_difference(idxs, test))
                    model_for_the_name.fit(df.loc[train, feature_names], df.loc[train, name])
                    predicted[test] = model_for_the_name.predict(df.loc[test, feature_names])
                    previous_divider = divider
                    divider += idxs.size // (cv + 1) + 1
                    divider = min(idxs.size, divider)
        elif crossval_typo == 'LOO arts':    # Leave One Out, per articles
            for name_art in kwargs['articles']:
                art = df.loc[df['PaperID'] == name_art]
                ind = np.where(df['PaperID'] != name_art)[0]
                descr_inds_art = del_from_list_if_not_in_df(art.T, idxs)
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
            try:
                df = pd.read_excel(f'{folder_to_save_results}/importance_data.xlsx', index_col='ind')
            except FileNotFoundError:
                df = pd.DataFrame()
            df['feature_' + kwargs["imp_name"]] = importance_data['feature_name']
            df['importance_' + kwargs["imp_name"]] = importance_data['importance']
            df.to_excel(f'{folder_to_save_results}/importance_data.xlsx', index_label='ind')

    if save_excel:
        output.loc[-1, :] = np.nan
        for name in target_names:
            output.loc[-1, name + '_predict'] = dict_r2[name]
        output.sort_index(inplace=True)
        output.to_excel(f'{folder_to_save_results}/{results_fname}', index=False)
        return f'{folder_to_save_results}/{results_fname}'
    if return_r2:
        return dict_r2


def fit_many_imputers_and_models(dataset: TiN_Dataset, target_names, feature_names, folder='', filename='ModelingResultsTable.xlsx',
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

    if not os.path.exists(folder):
        os.makedirs(folder)

    fill_value = keyargs['fill_value']
    del keyargs['fill_value']
    true_vs_pred_fname2 = true_vs_pred_fname = ''
    if 'true_vs_predicted_picture_fname' in keyargs:
        true_vs_pred_fname = keyargs['true_vs_predicted_picture_fname']
        del keyargs['true_vs_predicted_picture_fname']
    for imp in imputers:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recover_dataframe(dataset.df, feature_names, recovery_method=imp, fill_value=fill_value)
        for model_typo in models:
            model_regr, model_classifier = get_regressor_classifier(model_typo)
            if true_vs_pred_fname:
                true_vs_pred_fname2 = f'{true_vs_pred_fname}_{imp}_{model_typo}'
            dict_res_model = crosval_fit_for_each_target(recove_frame, dataset.dict_idxs, dataset.dict_norm,
                                                         target_names, feature_names, model_regr, model_classifier,
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


def fit_one_target_others_features(dataset: TiN_Dataset, target_names, feature_names, folder='', filename='ModelingResultsTable.xlsx',
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

    if not os.path.exists(folder):
        os.makedirs(folder)

    fill_value = keyargs['fill_value']
    del keyargs['fill_value']
    for imp in imputers:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recover_dataframe(dataset.df, feature_names + target_names, recovery_method=imp, fill_value=fill_value)
        for model_typo in models:
            model_regr, model_classifier = get_regressor_classifier(model_typo)
            for i, name in enumerate(target_names):
                dict_res_model = crosval_fit_for_each_target(
                    recove_frame, dataset.dict_idxs, dataset.dict_norm,
                    [name], feature_names + target_names[:i] + target_names[i + 1:],
                    model_regr, model_classifier,
                    imp_name=imp, **keyargs)
                if name in dict_res_model:
                    dict_res[name][j] = dict_res_model[name]
            j += 1
            print('Fitting completed')
    for name in target_names:
        res[name] = dict_res[name]
    res.T.to_excel(f'{folder}/{filename}')


def descr_analysis(dataset: TiN_Dataset, columns, dest_folder, number_of_tests=20, model_typo='ExtraTR', imputer_typo='const',
                   fill_value=-2, distribution: str = 'shuffle_origin'):
    if not isinstance(columns, np.ndarray):
        columns = np.array(columns)
    # print(data_columns)
    recovered_df = recover_dataframe(dataset.df, columns, recovery_method=imputer_typo, fill_value=fill_value)
    dict_idxs = dataset.dict_idxs
    # dict_norm = dataset.dict_norm

    if model_typo == 'ExtraTR':
        model_regressor = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_typo == 'RidgeCV':
        model_regressor = RidgeCV()
        model_classifier = LogisticRegression()
    else:
        assert False, 'model_typo for descr_analysis should be ExtraTR or RidgeCV'

    out_data = pd.DataFrame()
    out_data['descriptor'] = np.empty(columns.size, dtype='U25')
    out_data['r2_arr'] = np.zeros(columns.size)
    out_data['statistic'] = np.zeros(columns.size)

    for i, name in enumerate(columns):
        try:
            out_data.loc[i, 'descriptor'] = name
            if name in nominal_descr:
                # target_values = recovered_df.loc[dict_idxs[name], name] * dict_norm[name][1] + dict_norm[name][0]
                target_values = recovered_df.loc[dict_idxs[name], name]
                target_values = target_values.to_numpy().astype('int32')
                cross_val_arr = skl.model_selection.cross_val_score(model_classifier, recovered_df.loc[dict_idxs[name], columns[columns != name]], target_values, cv=min(10, len(dict_idxs[name])))
                # print(cross_val_arr)
                origin_r2 = np.mean(cross_val_arr)  # ???
                out_data.loc[i, 'r2_arr'] = origin_r2
            else:
                origin_arr = recovered_df.loc[dict_idxs[name], name]
                cross_val_arr = skl.model_selection.cross_val_predict(model_regressor, recovered_df.loc[dict_idxs[name], columns[columns != name]], recovered_df.loc[dict_idxs[name], name], cv=min(10, len(dict_idxs[name])))
                origin_r2 = scoreFast(origin_arr, cross_val_arr)  # R2 value
                out_data.loc[i, 'r2_arr'] = origin_r2
            random_r2 = np.empty(number_of_tests)
            for j in range(number_of_tests):
                if name in nominal_descr:
                    vals, prob = np.unique(recovered_df.loc[dict_idxs[name], name], return_counts=True)
                    # vals = vals * dict_norm[name][1] + dict_norm[name][0]
                    vals.astype('int32')
                    sz = len(recovered_df.loc[dict_idxs[name], name])
                    prob = prob / sz
                    ind = np.random.multinomial(n=1, pvals=prob, size=sz)
                    random_values = np.array([vals[np.where(ind[k])[0][0]] for k in range(sz)])
                    cross_val_arr = skl.model_selection.cross_val_score(model_classifier, recovered_df.loc[dict_idxs[name],
                                                columns[columns != name]], random_values, cv=min(10, len(dict_idxs[name])))
                    random_r2[j] = np.mean(cross_val_arr)
                else:
                    random_values = get_random_values(recovered_df.loc[dict_idxs[name], name], distribution=distribution)
                    cross_val_arr = skl.model_selection.cross_val_predict(model_regressor, recovered_df.loc[dict_idxs[name], columns[columns != name]], random_values, cv=min(10, len(dict_idxs[name])))
                    random_r2[j] = scoreFast(random_values, cross_val_arr)
            print(f'descriptor: {name}; criteria: {np.sum(origin_r2 > random_r2) / number_of_tests}')
            print(origin_r2, random_r2)
            out_data.loc[i, 'statistic'] = np.sum(origin_r2 > random_r2) / number_of_tests
        except FileNotFoundError:  # to unable exception
            print(f'Something went wrong! Descriptor: {name}')
    out_data.to_excel(f'{dest_folder}/descr_analysis.xlsx')


def article_analysis(dataset: TiN_Dataset, articles, data_columns, model_typo='ExtraTR', imp='const', fill_value=-2,
                     out_file_name='ArticleAnalysis.xlsx', out_folder='.',
                     predict_file_name='PredictTable.xlsx', predict_to_file=True, way_to_compute_score='mean'):
    df = dataset.df
    dict_idxs = dataset.dict_idxs
    dict_norm = dataset.dict_norm

    if not isinstance(data_columns, np.ndarray):
        data_columns = np.array(data_columns)
    predict_frame = copy.deepcopy(df)
    # print(data_columns)
    recove_frame = recover_dataframe(df, data_columns, recovery_method=imp, fill_value=fill_value)

    if model_typo == 'ExtraTR':
        model_regressor = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_typo == 'RidgeCV':
        model_regressor = RidgeCV()
        model_classifier = LogisticRegression()
    else:
        assert False, 'model_name for article_analysis should be ExtraTR or RidgeCV'

    dict_v = dict()
    for descr in data_columns:
        if descr in nominal_descr:
            if way_to_compute_score == 'relative':
                # target_values = recove_frame.loc[dict_idxs[descr], descr] * dict_norm[descr][1] + dict_norm[descr][0]
                target_values = recove_frame.loc[dict_idxs[descr], descr]
                target_values.astype('int32')
                if len(dict_idxs[descr]) >= 5:
                    cross_val_arr = skl.model_selection.cross_val_predict(model_classifier, recove_frame.loc[dict_idxs[descr], data_columns[data_columns != descr]], target_values, cv=min(5, len(dict_idxs[descr])))
                else:
                    cross_val_arr = np.full(len(dict_idxs[descr]), 0.5)
                v = np.sum(recove_frame.loc[dict_idxs[descr], descr] == cross_val_arr)/cross_val_arr.size
            elif way_to_compute_score == 'mean':
                _, frequency = np.unique(recove_frame.loc[dict_idxs[descr], descr], return_counts=True)
                v = np.max(frequency) / np.sum(frequency)
            else:
                assert False, 'way_to_compute_score parameter for article_analysis should be "relative" or "mean"'
        else:
            if way_to_compute_score == 'relative':
                if len(dict_idxs[descr]) >= 5:
                    cross_val_arr = skl.model_selection.cross_val_predict(model_regressor, recove_frame.loc[dict_idxs[descr], data_columns[data_columns != descr]], recove_frame.loc[dict_idxs[descr], descr], cv=min(5, len(dict_idxs[descr])))
                else:
                    cross_val_arr = np.full(len(dict_idxs[descr]), np.mean(recove_frame.loc[dict_idxs[descr], descr]))
                v = np.mean((recove_frame.loc[dict_idxs[descr], descr] - cross_val_arr)**2)
            elif way_to_compute_score == 'mean':
                v = np.mean((recove_frame.loc[dict_idxs[descr], descr] - np.mean(recove_frame.loc[dict_idxs[descr], descr]))**2)
            else:
                assert False, 'way_to_compute_score parameter for article_analysis should be "relative" or "mean"'
        dict_v[descr] = v
    dict_frequency = dict()
    for descr in del_from_list_if_not_in_df(df, nominal_descr):
        _, frequency = np.unique(recove_frame.loc[dict_idxs[descr], descr], return_counts=True)
        dict_frequency[descr] = np.max(frequency)/len(dict_idxs[descr])
    output = pd.DataFrame()
    output['descr'] = data_columns
    for name in articles:
        art = recove_frame.loc[recove_frame['PaperID'] == name]
        ind = np.where(recove_frame['PaperID'] != name)[0]
        scores = [None] * len(data_columns)
        for i, descr in enumerate(data_columns):
            if descr in nominal_descr:
                model_descr = model_classifier
            else:
                model_descr = model_regressor
            descr_inds_art = del_from_list_if_not_in_df(art.T, dict_idxs[descr])
            ind_notnull = np.intersect1d(dict_idxs[descr], ind)
            # print(recove_frame[descr].dtypes)
            # print(descr)
            # print(recove_frame[descr].tolist())
            if ind_notnull.size > 0 and len(descr_inds_art) > 0:
                # print(recove_frame.loc[ind_notnull, data_columns[data_columns != descr]].shape)
                # print(recove_frame.loc[ind_notnull, descr].shape)
                if descr in nominal_descr:
                    target_values = recove_frame.loc[dict_idxs[descr], descr] * dict_norm[descr][1] + dict_norm[descr][0]
                    target_values.astype('int32')
                else:
                    target_values = recove_frame.loc[dict_idxs[descr], descr]
                model_descr.fit(recove_frame.loc[ind_notnull, data_columns[data_columns != descr]], target_values.loc[ind_notnull])
                arr = model_descr.predict(art.loc[descr_inds_art, data_columns[data_columns != descr]])
                if descr in nominal_descr:
                    u = np.sum(target_values.loc[descr_inds_art] == arr)/arr.size
                    # fr = dict_frequency[descr]
                    # score_val = 1 - (1 - u) / (1 - dict_v[descr])
                    # score = f'{u:.2} {dict_v[descr]:.2} {fr:.2} {score_val:.2}'
                    score = u
                else:
                    u = np.mean((art.loc[descr_inds_art, descr] - arr)**2)
                    score = 1 - u/dict_v[descr]
                    # score = scoreByConstPredict(np.array(art.loc[descr_inds_art, descr]), arr, np.mean(recove_frame.loc[ind_notnull, descr]))
                    # if score < -1e3:
                    #     print(score)
                    #     print(arr)
                    #     print(np.array(art.loc[descr_inds_art, descr]))
                    #     exit(0)
                scores[i] = score
                if predict_to_file:
                    if descr in str_descr:
                        enc = LabelEncoder()
                        enc.classes_ = dataset.dict_labels[descr]
                        predict_frame.loc[descr_inds_art, descr] = enc.inverse_transform(arr)
                    else:
                        # predict_frame.loc[descr_inds_art, descr] = arr * dict_norm[descr][1] + dict_norm[descr][0]
                        predict_frame.loc[descr_inds_art, descr] = arr
        output[name] = scores
        print('Progress!')
    file_name = out_file_name[:out_file_name.rfind('.')] + '_' + way_to_compute_score + '.xlsx'
    output.to_excel(f'{out_folder}/{file_name}')
    if predict_to_file:
        predict_frame.T.to_excel(predict_file_name)


def qheatmap_data(dataset: TiN_Dataset, dest_folder, target_name, feature_names, model_typo='ExtraTR', recovering='const'):
    recove_frame = recover_dataframe(dataset.df, feature_names, recovery_method=recovering, fill_value=FILL_VALUE)
    # true_values = df.loc[dataset.dict_dxs[descr], descr]
    # v = np.mean((true_values - np.mean(true_values)) ** 2)
    matrix_r2 = np.zeros((feature_names.size, feature_names.size))
    model_regr, model_classifier = get_regressor_classifier(model_typo)
    for i in range(feature_names.size):
        for j in range(i, feature_names.size):
            f1 = feature_names[i]
            f2 = feature_names[j]
            if f1 == f2:
                dict_value = crosval_fit_for_each_target(recove_frame, dataset.dict_idxs, dataset.dict_norm,
                                                         [target_name], [f1], model_regr, model_classifier,
                                                         crossval_typo='3:1', imp_name='')
            else:
                dict_value = crosval_fit_for_each_target(recove_frame, dataset.dict_idxs, dataset.dict_norm,
                                                         [target_name], [f1, f2], model_regr, model_classifier,
                                                         crossval_typo='3:1', imp_name='')
            matrix_r2[i][j] = dict_value[target_name]
            if f1 != f2:
                matrix_r2[j][i] = dict_value[target_name]
    out_df = pd.DataFrame()
    for ind, name in enumerate(feature_names):
        out_df[name] = matrix_r2[ind]
    out_df.to_excel(f'{dest_folder}/qheatmap_data.xlsx')


FOLDER = f'./221214'
FILL_VALUE = -2
USE_ENCODER = True


def run_unfiltered():
    # DATA READING
    # CHANGE item 'last' in columns_to_read if number of samples has changed
    # data_TiN = TiN_Dataset('./DataTable_Guda_2.xlsx', 139, columns_to_read={'last': 'EN'}, skiprows=1)
    data_TiN = TiN_Dataset('./DataTable_Guda_3.xlsx', 281, columns_to_read={'last': 'JZ'}, skiprows=1)

    # FILTERING
    data_TiN.delete_empty_invalid_names()
    data_TiN.apply_filter(threshold_for_descrs=0., filter_samples=False)  # Filter only completely invalid descriptors

    # INFORMATION EXTRACTION
    articles_names = data_TiN.get_articles_names()
    data_TiN.remember_not_missed_for_each_descr()

    # ENCODING, NORMALIZING
    str_descrs = del_from_list_if_not_in_df(data_TiN.df, str_descr)
    data_TiN.apply_LabelEncoder(str_descrs)
    data_TiN.normalize(norm_nominal=False)

    # FITTING
    dest_folder = f'{FOLDER}/unfiltered'
    os.makedirs(dest_folder, exist_ok=True)
    fit_many_imputers_and_models(data_TiN, ['H'], del_from_list_if_not_in_df(data_TiN.df, exp_descr), fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=False, folder=dest_folder, articles=articles_names, true_vs_predicted_picture_fname='')

    # PLOTTING
    df = data_TiN.df
    plot_folder = f'{dest_folder}/plots'
    os.makedirs(plot_folder, exist_ok=True)
    count_sparsity_plotting_bar(df.loc[:, lists_difference(data_TiN.df.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                                out_file_path=f'{plot_folder}/fig2{EXT}')
    descr_sparcity_table(data_TiN, df.columns, all_table=True, out_folder=plot_folder)


def run_filtered():
    # HARD FILTER

    # DATA READING
    # CHANGE item 'last' in columns_to_read if number of samples has changed
    # data_TiN = TiN_Dataset('./DataTable_Guda_2.xlsx', 139, columns_to_read={'last': 'EN'}, skiprows=1)
    data_TiN = TiN_Dataset('./DataTable_Guda_3.xlsx', 281, columns_to_read={'last': 'JZ'}, skiprows=1)

    # FILTERING
    data_TiN.delete_empty_invalid_names()
    data_TiN.apply_filter(threshold_for_descrs=5.1, filter_samples=True)

    # INFORMATION EXTRACTION
    articles_names = data_TiN.get_articles_names()
    data_TiN.remember_not_missed_for_each_descr()

    # ENCODING, NORMALIZING
    str_descrs = del_from_list_if_not_in_df(data_TiN.df, str_descr)
    data_TiN.apply_LabelEncoder(str_descrs)
    data_TiN.normalize(norm_nominal=False)

    # FITTING
    dest_folder = f'{FOLDER}/filtered'
    os.makedirs(dest_folder, exist_ok=True)
    fit_many_imputers_and_models(data_TiN, ['H'], del_from_list_if_not_in_df(data_TiN.df, exp_descr), fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=True, folder=dest_folder, articles=articles_names, true_vs_predicted_picture_fname=f'true_vs_predicted')

    qheatmap_data(data_TiN, target_name='H', feature_names=np.array(best_features),
                  dest_folder=dest_folder, model_typo='ExtraTR', recovering='const')

    # PLOTTING
    df = data_TiN.df
    plot_folder = f'{dest_folder}/plots'
    os.makedirs(plot_folder, exist_ok=True)
    count_sparsity_plotting_bar(df.loc[:, lists_difference(data_TiN.df.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                                out_file_path=f'{plot_folder}/fig2{EXT}')
    descr_sparcity_table(data_TiN, df.columns, all_table=True, out_folder=plot_folder)

    importance_bars(out_folder=plot_folder,
                    in_path=f'{dest_folder}/importance_data.xlsx', name='H')
    quality_heatmap(dest_folder, plot_folder)

    training_results_bar(f'{FOLDER}/unfiltered/ModelingResultsTable.xlsx', ['H'], plot_folder,
                         one_more_file_path=f'{FOLDER}/filtered/ModelingResultsTable.xlsx',
                         out_file_name='fig3',
                         add_text_plot=[(0.48, 0.95, 'ExtraTrees'), (0.7, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
                         text_plot_ops={'transform': True},)


def descrs_sparsity_to_file():
    data_TiN = TiN_Dataset('./DataTable_Guda_3.xlsx', 281, columns_to_read={'last': 'JZ'}, skiprows=1)
    data_TiN.get_descrs_samples_sparsity(dest_filepath_descrs=f'{FOLDER}/experiments/sparsity_full_data.txt')


def run_descr_analysis():
    data_TiN = TiN_Dataset('./DataTable_Guda_3.xlsx', 281, columns_to_read={'last': 'JZ'}, skiprows=1)

    data_TiN.delete_empty_invalid_names()
    data_TiN.apply_filter(threshold_for_descrs=5.1, filter_samples=True)

    data_TiN.remember_not_missed_for_each_descr()

    str_descrs = del_from_list_if_not_in_df(data_TiN.df, str_descr)
    data_TiN.apply_LabelEncoder(str_descrs)
    data_TiN.normalize(norm_nominal=False)

    descr_analysis(data_TiN, data_TiN.df.columns[data_TiN.df.columns != 'PaperID'], dest_folder=f'{FOLDER}/descr_analysis')


def run_filtered_Berkovich_only():
    data_TiN = TiN_Dataset('./DataTable_Guda_3.xlsx', 281, columns_to_read={'last': 'JZ'}, skiprows=1)

    data_TiN.apply_filter(threshold_for_descrs=5.1, filter_samples=True)
    data_TiN.apply_filter_descr_values_ranges(IndentMethod='Berkovich')
    data_TiN.delete_empty_invalid_names()

    # data_TiN.df.to_excel(f'{FOLDER}/Berkovich/Check.xlsx')

    articles_names = data_TiN.get_articles_names()
    data_TiN.remember_not_missed_for_each_descr()

    str_descrs = del_from_list_if_not_in_df(data_TiN.df, str_descr)
    data_TiN.apply_LabelEncoder(str_descrs)
    data_TiN.normalize(norm_nominal=False)

    # FITTING
    dest_folder = f'{FOLDER}/Berkovich'
    os.makedirs(dest_folder, exist_ok=True)
    fit_many_imputers_and_models(data_TiN, ['H'], del_from_list_if_not_in_df(data_TiN.df, exp_descr), fill_value=FILL_VALUE, crossval_typo='3:1', count_importance_to_file=True, folder=dest_folder, articles=articles_names, true_vs_predicted_picture_fname=f'true_vs_predicted')

    # PLOTTING
    df = data_TiN.df
    plot_folder = f'{dest_folder}/plots'
    os.makedirs(plot_folder, exist_ok=True)
    count_sparsity_plotting_bar(df.loc[:, lists_difference(data_TiN.df.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                                out_file_path=f'{plot_folder}/fig2{EXT}')
    descr_sparcity_table(data_TiN, df.columns, all_table=True, out_folder=plot_folder)

    importance_bars(out_folder=plot_folder,
                    in_path=f'{dest_folder}/importance_data.xlsx', name='H')

    training_results_bar(f'{FOLDER}/Berkovich/ModelingResultsTable.xlsx', ['H'], plot_folder,
                         one_more_file_path=None,
                         out_file_name='fig3',
                         add_text_plot=[(0.5, 0.95, 'ExtraTrees'), (0.7, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
                         text_plot_ops={'transform': True},)


def main():
    run_unfiltered()
    run_filtered()

    # descrs_sparsity_to_file()

    # run_descr_analysis()

    # run_filtered_Berkovich_only()

    pass


if __name__ == '__main__':
    main()
