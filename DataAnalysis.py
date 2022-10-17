# import \
#     copy
# import \
#     itertools

# import sys

# import numpy as np
# import pandas as pd
import \
    os

import sklearn as skl
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute._iterative import IterativeImputer
from sklearn.linear_model import RidgeCV, LogisticRegression
# from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.svm import SVC, SVR
# from fancyimpute import Solver, NuclearNormMinimization, MatrixFactorization, IterativeSVD, SimpleFill, SoftImpute, BiScaler, KNN, SimilarityWeightedAveraging
# from datawig import SimpleImputer as DwImputer

from pyfitit import *

# from  usable_functions_1 import *
from TiN_frame_process import *
from TiN_plots import *


def model_create(model_str: str):
    """
    The function receives model name
    and returns two objects: regressor object, classifier object

    :param model_str:
    :return:
    """
    if model_str == 'RidgeCV':
        model = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
        model_classifier = LogisticRegression()
    elif model_str == 'ExtraTR':
        model = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_str == 'Gauss':
        kernel = C(1.0) * RBF(1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model_classifier = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=0)
    elif model_str == 'SVM':
        model = SVR(kernel='rbf', C=10, epsilon=0.1)
        model_classifier = SVC(kernel='rbf', C=10)
    else:
        assert False, f'invalid value for create_method: {model_str}'
    return model, model_classifier


def model_create_frame_fit(frame, target_columns, used_columns, model_str='RidgeCV', folder='',
                           data_set='original', crossval_mode='LOO arts', mod='r2score', draw_picture=False, count_importance=False, **key_args):
    if not isinstance(target_columns, list):
        target_columns = list(target_columns)
    if not isinstance(used_columns, list):
        used_columns = list(used_columns)
    num_expr = frame.shape[0]
    model, model_classifier = model_create(model_str)
    dict_r2 = dict()
    if mod == 'all':
        output = pd.DataFrame()
    # arr_score = np.array([-1] * num_expr)
    # k = 0
    for name in target_columns:
        if name in str_descr:
            model_name = model_classifier
        else:
            model_name = model
        if data_set == 'original':
            inds = dict_ind[name]
        elif data_set == 'all':
            inds = [i for i in range(num_expr)]
        else:
            assert False, f'invalid value for data_set: {data_set}'
        original_values = np.array([None] * num_expr)
        values = np.array([None] * num_expr)
        # print(dataframe[name].dtypes)
        # print(name)
        # print(dataframe[name].tolist())
        # exit(0)
        original_values[inds] = frame.loc[inds, name]
        count_score = True
        if crossval_mode == 'LOO exps':
            for i in range(len(inds)):
                model_name.fit(frame.loc[del_per_index(inds, i), used_columns], frame.loc[del_per_index(inds, i), name])
                arr = model_name.predict(frame.loc[inds[i], used_columns].to_numpy().reshape(1, -1))
                assert arr.size == 1
                values[inds[i]] = arr[0]
                # if model_str == 'RidgeCV':
                #     print(model.alpha_)
        elif crossval_mode == crossval_mode[0] + ':1':
            assert crossval_mode[0] in '3456789', f'invalid value for crossval_mode: {crossval_mode}'
            cv = int(crossval_mode[0])
            if len(inds) < (cv + 1) * 5:
                print(f'Warning! Number of experiments is too small for {name}')
                values[inds] = np.mean(original_values[inds]) - 1
            else:
                inds = np.array(inds)
                np.random.shuffle(inds)
                previous_divider = 0
                divider = inds.size // (cv + 1) + 1
                for i in range(cv + 1):
                    test = inds[previous_divider:divider]
                    train = np.array(remove_many(inds, test))
                    model_name.fit(frame.loc[train, used_columns], frame.loc[train, name])
                    values[test] = model_name.predict(frame.loc[test, used_columns])
                    previous_divider = divider
                    divider += inds.size // (cv + 1) + 1
                    divider = min(inds.size, divider)
        elif crossval_mode == 'LOO arts':
            for name_art in key_args['articles']:
                art = frame.loc[frame['PaperID'] == name_art]
                ind = np.where(frame['PaperID'] != name_art)[0]
                if name in str_descr:
                    model_descr = model_classifier
                else:
                    model_descr = model
                descr_inds_art = clean_list(art.T, inds, inplace=False)
                ind_notnull = np.intersect1d(inds, ind)
                if (ind_notnull.size > 0) and (len(descr_inds_art) > 0):
                    model_descr.fit(frame.loc[ind_notnull, used_columns], frame.loc[ind_notnull, name])
                    values[descr_inds_art] = model_descr.predict(art.loc[descr_inds_art, used_columns])
                    # if model_str == 'RidgeCV':
                    #     print(model.alpha_)
                elif ind_notnull.size == 0:
                    count_score = False
        else:
            assert False, f'invalid value for crossval_mode: {crossval_mode}'
        if mod == 'all':
            if name in str_descr:
                output[name + '_original'] = original_values
                output[name + '_predict'] = values
            else:
                original_values[inds] = original_values[inds] * dict_norm[name][1] + dict_norm[name][0]
                output[name + '_original'] = original_values
                values[inds] = values[inds] * dict_norm[name][1] + dict_norm[name][0]
                output[name + '_predict'] = values
                abs_error = copy.deepcopy(values)
                abs_error[inds] = abs(values[inds] - original_values[inds])
                output[name + '_absolute_error'] = abs_error
        if count_score:
            dict_r2[name] = scoreFast(original_values[inds], values[inds])
        else:
            dict_r2[name] = np.nan
        # k += 1
        if draw_picture:
            if mod != 'all':
                if name not in str_descr:
                    original_values[inds] = original_values[inds] * dict_norm[name][1] + dict_norm[name][0]
                    values[inds] = values[inds] * dict_norm[name][1] + dict_norm[name][0]
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.scatter(values, original_values, c='red')
            ax.plot([-10, 100], [-10, 100], c='red')
            ax.set_xlabel(f'{name}_predicted')
            ax.set_ylabel(f'{name}_original')
            fig.savefig(f'{folder}{name}_prediction_picture_{model_str}.png')
            plt.close(fig)
        if model_str == 'ExtraTR' and count_importance:
            importance_data = np.empty(len(used_columns), dtype=[('feature_name', 'U20'), ('importance', float)])
            for i in range(len(used_columns)):
                importance_data[i] = used_columns[i], model_name.feature_importances_[i]
            importance_data.sort(order='importance')
            try:
                df = pd.read_excel(f'{folder}importance_data.xlsx')
                del df['Unnamed: 0']
            except FileNotFoundError:
                df = pd.DataFrame()
            df['feature_' + key_args["imp_name"]] = importance_data['feature_name']
            df['importance_' + key_args["imp_name"]] = importance_data['importance']
            df.to_excel(f'{folder}importance_data.xlsx')
            # fig, ax = plt.subplots(1, figsize=MAIN_FIG_SIZE)
            # ax.tick_params(labelsize=8)
            # ax.barh(importance_data['feature_name'], importance_data['importance'],
            #         color=color_arr_from_arr(model_name.feature_importances_, init_color=(1.0, 1.0, 1.0), finish_color=(1.0, 0.9, 0.4), bottom=-0.1))
            # ax.barh(used_columns, model_name.feature_importances_, color='orange')
            # # ax.tick_params(labelrotation=45)
            # ax.set_title(f'Feature importance {name}', fontsize=MAIN_PLOT_FONT_SIZE)
            # fig.savefig(f'feature_importance_{key_args["imp_name"]}_{name}.png')
            # plt.close(fig)
    if mod == 'all':
        output.loc[-1, :] = np.nan
        for name in target_columns:
            output.loc[-1, name + '_predict'] = dict_r2[name]
        output.sort_index(inplace=True)
        output.to_excel(f'{folder}ResultModeling.xlsx', index=False)
    elif mod == 'r2score':
        return dict_r2


def get_result(dataframe, target_columns, used_columns, folder='', **keyargs):

    """

    Each column from target columns is predicted
    using used columns

    :param dataframe:
    :param target_columns:
    :param used_columns:
    :param folder:
    :param keyargs:
    :return:
    """

    # imp_mass = ['simple', 'iterative', 'knn', 'soft_imp', 'const', 'matr_factoriz', 'similarity']
    imp_mass = ['const', 'simple', 'iterative', 'knn']
    # model_mass = ['ExtraTR', 'RidgeCV', 'Gauss', 'SVM']
    model_mass = ['ExtraTR', 'SVM', 'RidgeCV']
    # model_mass = ['ExtraTR']
    res = pd.DataFrame()
    frame_imps_mass = []
    for name in imp_mass:
        frame_imps_mass += [name] * len(model_mass)
    res['imputer'] = np.array(frame_imps_mass)
    res['model'] = np.array(model_mass * len(imp_mass))
    dict_res = dict()
    for name in target_columns:
        dict_res[name] = np.array([None] * (len(imp_mass) * len(model_mass)))
    j = 0
    for imp in imp_mass:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recovery_data(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'])
        # print(recove_frame.dtypes)
        for model in model_mass:
            dict_res_model = model_create_frame_fit(recove_frame, target_columns, used_columns, model_str=model,
                                                    crossval_mode=keyargs['crossval_mode'], articles=keyargs['articles'],
                                                    draw_picture=keyargs['draw_picture'],
                                                    count_importance=keyargs['count_importance'], imp_name=imp,
                                                    folder=folder,
                                                    out_folder_picture='Scatters/Scatters_predict/table3/')
            for name in dict_res_model:
                dict_res[name][j] = dict_res_model[name]
            j += 1
            print('Yes!')
    for name in target_columns:
        res[name] = dict_res[name]
    res.T.to_excel(f'{folder}ModelingResults.xlsx')


def get_result_one_target_per_all_others(dataframe, target_columns, used_columns, folder='', **keyargs):

    """

    Each column from target columns column is predicted using
    used_columns and other target columns

    :param dataframe:
    :param target_columns:
    :param used_columns:
    :param folder:
    :param keyargs:
    :return:
    """

    # imp_mass = ['simple', 'iterative', 'knn', 'soft_imp', 'const', 'matr_factoriz', 'similarity']
    imp_mass = ['const', 'simple', 'iterative', 'knn']
    # model_mass = ['ExtraTR', 'RidgeCV', 'Gauss', 'SVM']
    model_mass = ['ExtraTR', 'RidgeCV', 'SVM']
    res = pd.DataFrame()
    frame_imps_mass = []
    for name in imp_mass:
        frame_imps_mass += [name] * len(model_mass)
    res['imputer'] = np.array(frame_imps_mass)
    res['model'] = np.array(model_mass * len(imp_mass))
    dict_res = dict()
    for name in target_columns:
        dict_res[name] = np.array([None] * (len(imp_mass) * len(model_mass)))
    j = 0
    for imp in imp_mass:
        # recove_frame = recove_and_normalize(dataframe, used_columns, recovery_method=imp, fill_value=keyargs['fill_value'], norm_nominal=True)
        recove_frame = recovery_data(dataframe, used_columns + target_columns, recovery_method=imp, fill_value=keyargs['fill_value'])
        # print(recove_frame.dtypes)
        for model in model_mass:
            for i, name in enumerate(target_columns):
                # print(name)
                dict_res_model = model_create_frame_fit(
                    recove_frame,
                    [name],
                    used_columns + target_columns[:i] + target_columns[i+1:],
                    model_str=model,
                    crossval_mode=keyargs['crossval_mode'], articles=keyargs['articles'],
                    draw_picture=keyargs['draw_picture'],
                    count_importance=keyargs['count_importance'], imp_name=imp,
                    out_folder_picture='Scatters/Scatters_predict/table3/')
                if name in dict_res_model:
                    dict_res[name][j] = dict_res_model[name]
            j += 1
            print('Yes!')
    for name in target_columns:
        res[name] = dict_res[name]
    res.T.to_excel(f'{folder}ModelingResults.xlsx')


def article_analysis(dataframe, articles, data_columns, model_name='ExtraTR', imp='const', fill_value=-2,
                     out_file_name='ArticleAnalysis.xlsx', out_folder='.',
                     predict_file_name='PredictTable.xlsx', predict_to_file=True, score_mod='mean'):
    if not isinstance(data_columns, np.ndarray):
        data_columns = np.array(data_columns)
    predict_frame = copy.deepcopy(dataframe)
    # print(data_columns)
    recove_frame = recovery_data(dataframe, data_columns, recovery_method=imp, fill_value=fill_value)
    if model_name == 'ExtraTR':
        model = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_name == 'RidgeCV':
        model = RidgeCV()
        model_classifier = LogisticRegression()
    else:
        assert False, 'model_name for article_analysis should be ExtraTR or RidgeCV'
    dict_v = dict()
    for descr in data_columns:
        if descr in nominal_descr:
            if score_mod == 'relative':
                target_values = recove_frame.loc[dict_ind[descr], descr] * dict_norm[descr][1] + dict_norm[descr][0]
                target_values.astype('int32')
                if len(dict_ind[descr]) >= 5:
                    cross_val_arr = skl.model_selection.cross_val_predict(model_classifier, recove_frame.loc[dict_ind[descr], data_columns[data_columns != descr]], target_values, cv=min(5, len(dict_ind[descr])))
                else:
                    cross_val_arr = np.full(len(dict_ind[descr]), 0.5)
                v = np.sum(recove_frame.loc[dict_ind[descr], descr] == cross_val_arr)/cross_val_arr.size
            elif score_mod == 'mean':
                _, frequency = np.unique(recove_frame.loc[dict_ind[descr], descr], return_counts=True)
                v = np.max(frequency) / np.sum(frequency)
            else:
                assert False, 'score_mod for article_analysis should be relative or mean'
        else:
            if score_mod == 'relative':
                if len(dict_ind[descr]) >= 5:
                    cross_val_arr = skl.model_selection.cross_val_predict(model, recove_frame.loc[dict_ind[descr], data_columns[data_columns != descr]], recove_frame.loc[dict_ind[descr], descr], cv=min(5, len(dict_ind[descr])))
                else:
                    cross_val_arr = np.full(len(dict_ind[descr]), np.mean(recove_frame.loc[dict_ind[descr], descr]))
                v = np.mean((recove_frame.loc[dict_ind[descr], descr] - cross_val_arr)**2)
            elif score_mod == 'mean':
                v = np.mean((recove_frame.loc[dict_ind[descr], descr] - np.mean(recove_frame.loc[dict_ind[descr], descr]))**2)
            else:
                assert False, 'score_mod for article_analysis should be relative or mean'
        dict_v[descr] = v
    dict_frequency = dict()
    for descr in clean_list(dataframe, nominal_descr, inplace=False):
        _, frequency = np.unique(recove_frame.loc[dict_ind[descr], descr], return_counts=True)
        dict_frequency[descr] = np.max(frequency)/len(dict_ind[descr])
    output = pd.DataFrame()
    output['descr'] = data_columns
    for name in articles:
        art = recove_frame.loc[recove_frame['PaperID'] == name]
        ind = np.where(recove_frame['PaperID'] != name)[0]
        mass = [None] * len(data_columns)
        for id, descr in enumerate(data_columns):
            if descr in nominal_descr:
                model_descr = model_classifier
            else:
                model_descr = model
            descr_inds_art = clean_list(art.T, dict_ind[descr], inplace=False)
            ind_notnull = np.intersect1d(dict_ind[descr], ind)
            # print(recove_frame[descr].dtypes)
            # print(descr)
            # print(recove_frame[descr].tolist())
            if ind_notnull.size > 0 and len(descr_inds_art) > 0:
                # print(recove_frame.loc[ind_notnull, data_columns[data_columns != descr]].shape)
                # print(recove_frame.loc[ind_notnull, descr].shape)
                if descr in nominal_descr:
                    target_values = recove_frame.loc[dict_ind[descr], descr] * dict_norm[descr][1] + dict_norm[descr][0]
                    target_values.astype('int32')
                else:
                    target_values = recove_frame.loc[dict_ind[descr], descr]
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
                mass[id] = score
                if predict_to_file:
                    if descr in str_descr:
                        enc = LabelEncoder()
                        enc.classes_ = dict_labels[descr]
                        predict_frame.loc[descr_inds_art, descr] = enc.inverse_transform(arr)
                    else:
                        predict_frame.loc[descr_inds_art, descr] = arr * dict_norm[descr][1] + dict_norm[descr][0]
        output[name] = mass
        print('Progress!')
    file_name = out_file_name[:out_file_name.rfind('.')] + '_' + score_mod + '.xlsx'
    output.to_excel(f'{out_folder}/{file_name}')
    if predict_to_file:
        predict_frame.T.to_excel(predict_file_name)


def descr_analysis(frame, columns, number_of_tests=20, model_name='ExtraTR', imp='const',
                   fill_value=-2, same_distribution=False):
    if not isinstance(columns, np.ndarray):
        columns = np.array(columns)
    # print(data_columns)
    recove_frame = recovery_data(frame, columns, recovery_method=imp, fill_value=fill_value)
    if model_name == 'ExtraTR':
        model = ExtraTreesRegressor(random_state=0)
        model_classifier = ExtraTreesClassifier(random_state=0)
    elif model_name == 'RidgeCV':
        model = RidgeCV()
        model_classifier = LogisticRegression()
    else:
        assert False, 'model_name for descr_analysis should be ExtraTR or RidgeCV'
    graph_data = pd.DataFrame()
    graph_data['descriptor'] = np.empty(columns.size, dtype='U25')
    graph_data['r2_arr'] = np.zeros(columns.size)
    graph_data['statistic'] = np.zeros(columns.size)
    i_1 = 0
    for name in columns:
        try:
            graph_data.loc[i_1, 'descriptor'] = name
            if name in nominal_descr:
                target_values = recove_frame.loc[dict_ind[name], name] * dict_norm[name][1] + dict_norm[name][0]
                target_values.astype('int32')
                cross_val_arr = skl.model_selection.cross_val_score(model_classifier, recove_frame.loc[dict_ind[name], columns[columns != name]], target_values, cv=min(10, len(dict_ind[name])))
                # print(cross_val_arr)
                origin_r2 = np.mean(cross_val_arr)  # R2 value
                graph_data.loc[i_1, 'r2_arr'] = origin_r2
            else:
                origin_arr = recove_frame.loc[dict_ind[name], name]
                cross_val_arr = skl.model_selection.cross_val_predict(model, recove_frame.loc[dict_ind[name], columns[columns != name]], recove_frame.loc[dict_ind[name], name], cv=min(10, len(dict_ind[name])))
                origin_r2 = scoreFast(origin_arr, cross_val_arr)  # R2 value
                graph_data.loc[i_1, 'r2_arr'] = origin_r2
            random_r2 = np.empty(number_of_tests)
            for i in range(number_of_tests):
                if name in nominal_descr:
                    vals, prob = np.unique(recove_frame.loc[dict_ind[name], name], return_counts=True)
                    vals = vals * dict_norm[name][1] + dict_norm[name][0]
                    vals.astype('int32')
                    sz = len(recove_frame.loc[dict_ind[name], name])
                    prob = prob/sz
                    ind = np.random.multinomial(n=1, pvals=prob, size=sz)
                    random_values = np.array([vals[np.where(ind[k])[0][0]] for k in range(sz)])
                    cross_val_arr = skl.model_selection.cross_val_score(model_classifier, recove_frame.loc[dict_ind[name],
                                                columns[columns != name]], random_values, cv=min(10, len(dict_ind[name])))
                    random_r2 = np.mean(cross_val_arr)
                else:
                    random_values = get_random_values(recove_frame.loc[dict_ind[name], name], distribution='shuffle_origin')
                    cross_val_arr = skl.model_selection.cross_val_predict(model, recove_frame.loc[dict_ind[name], columns[columns != name]], random_values, cv=min(10, len(dict_ind[name])))
                    random_r2[i] = scoreFast(random_values, cross_val_arr)
            print(name, np.sum(origin_r2 > random_r2) / number_of_tests)
            print(origin_r2, random_r2)
            graph_data.loc[i_1, 'statistic'] = np.sum(origin_r2 > random_r2) / number_of_tests
            i_1 += 1
        except Exception:
            print(f'Something went wrong! Name{name}')
    graph_data.to_excel('descr_analysis.xlsx')


def get_quality_1_2_3_4(frame, used_cols, target_descr, model_name, flags=(1, 1, 1, 0, 0), cv_parts=5, out_folder='', **kwargs):
    assert isinstance(target_descr, str), 'parameter descr should be string'
    assert kwargs['recovery_method'] != 'iterative', 'only not iterative for this function'
    model_regr, model_class = model_create(model_name)
    features = used_cols
    try:
        features.remove(target_descr)
    except ValueError:
        print(f'{target_descr} not in cols')
    for i in range(1, len(flags) + 1):
        if flags[i - 1]:
            descriptor.descriptor_quality(recovery_data(frame.loc[dict_ind[target_descr], :],
                                                        used_cols, recovery_method=kwargs['recovery_method'], fill_value=kwargs['fill_value']),
                                          [target_descr], features, model_regr=model_regr, model_class=model_class,
                                          feature_subset_size=i, cv_repeat=10, cv_parts_count=cv_parts,
                                          folder=f'{out_folder}quality_{i}', shuffle=True)
            print('all right')


def get_frame_of_ratios(frame, articles, num_rows_per_art=10, mode='ratio'):
    frame.to_excel('x_file_1.xlsx')
    out = pd.DataFrame(np.full(frame.shape, np.nan))
    out.columns = frame.columns
    recove_frame = recovery_data(frame, frame.columns, recovery_method='const', fill_value=1)
    ind_row = 0
    cols = remove_many(recove_frame.columns, nominal_descr + ['PaperID'])
    nominal_cols = clean_list(recove_frame, nominal_descr, inplace=False)
    if mode == 'ratio':
        for name in articles:
            art = recove_frame.loc[recove_frame['PaperID'] == name, cols]
            nominal_part = recove_frame.loc[recove_frame['PaperID'] == name, nominal_cols]
            num_exp = art.shape[0]
            if num_exp > 1:
                art = art.to_numpy()
                nominal_part = nominal_part.to_numpy()
                allinds = utils.comb_index(num_exp, 2, repetition=False)
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
                allinds = utils.comb_index(num_exp, 2, repetition=False)
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


def qheatmap_data(frame, folder, descr, features, model='ExtraTR', recovering='const'):
    recove_frame = recovery_data(frame, features, recovery_method=recovering, fill_value=FILL_VALUE)
    # true_values = frame.loc[dict_ind[descr], descr]
    # v = np.mean((true_values - np.mean(true_values)) ** 2)
    matrix_r2 = np.zeros((features.size, features.size))
    for i in range(features.size):
        for j in range(i, features.size):
            f1 = features[i]
            f2 = features[j]
            if f1 == f2:
                dict_value = model_create_frame_fit(recove_frame, [descr], [f1], model_str=model,
                                                    crossval_mode='3:1', imp_name='')
            else:
                dict_value = model_create_frame_fit(recove_frame, [descr], [f1, f2], model_str=model,
                                                    crossval_mode='3:1', imp_name='')
            matrix_r2[i][j] = dict_value[descr]
            if f1 != f2:
                matrix_r2[j][i] = dict_value[descr]
    df = pd.DataFrame()
    for ind, name in enumerate(features):
        df[name] = matrix_r2[ind]
    df.to_excel(f'{folder}qheatmap_data.xlsx')


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
#         frame = check_code(frame, main_chance=0.2)
#     pass


# IMPORTANT!!!
# many of global lists, dicts
# are currently in file 'TiN_frame_process.py'

dict_last_cols['_Guda_2'] = 'EN'
dict_num_exps['_Guda_2'] = 139

dict_last_cols['_Guda_3'] = 'JZ'
dict_num_exps['_Guda_3'] = 281

input_folder_prefix = ''
input_postfix = '_Guda_3'
folder_prefix = 'result/Predict_ExtraTR/'
postfix = '.from1to3'

file_input_data = f'{input_folder_prefix}DataTable{input_postfix}.xlsx'

descrs = pd.read_excel(file_input_data, usecols='C', skiprows=1).to_numpy().reshape(1, -1)[0]
# 0.1: soft filter; 2.1: hard filter
# values in article: -2. for fig1, S1, S2; 2.1 for fig2; 0.1, 2.1 for fig3, but mostly used prepared data;
# 2.1 for fig 4, 5, but mostly used prepared data;
filter_rubbish = (pd.read_excel(file_input_data, usecols='E', skiprows=1) > -1.1).to_numpy().reshape(1, -1)[0]
good = (pd.read_excel(file_input_data, usecols='E', skiprows=1) > 2.1).to_numpy().reshape(1, -1)[0]
good_descrs = descrs[good]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# поменять диапозон столбцов если изменится число экспериментов
x = pd.read_excel(file_input_data, usecols=f'F:{dict_last_cols[input_postfix]}', skiprows=1)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

assert len(x.columns) == dict_num_exps[input_postfix], 'Check the number of experiments in your table! ' + str(len(x.columns))

x = x.T
x.reset_index(drop=True, inplace=True)

x.columns = descrs
good_exp_inds = np.arange(x.shape[0])[x['Bad'].isna()]

# !!! не забудь раскомменировать строчки 593-597
# degree of main_chance:
# high: 0.5
# medium: 0.3
# small: 0.2
# very_small: 0.08
# x = check_code(x, main_chance=0.2)

# x.to_excel('x_file.xlsx')
# exit(0)

# arts_descrs_picture(x, hist_or_bar='bar', out_folder=f'{PLOT_FOLDER}/')  # figS1

PLOT_FOLDER = './221010_pictures/test_all_plots'
if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)
all_plots(get_filtered_frame(x, True, False, filter_rubbish, None), PLOT_FOLDER,
          filter_ops=(True, True, good_descrs, good_exp_inds),
          results_ops={'data_file_path': '22_04_results/unfiltered/ModelingResults.xlsx',
                       'one_more_file_path': '22_04_results/filtered/ModelingResults.xlsx',
                       'out_file_name': 'fig3',
                       'bar_descrs': clean_list(x, ['H'], inplace=False),
                       'add_text_plot': [(0.56, 0.95, 'ExtraTrees'), (0.72, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
                       'text_plot_ops': {'transform': True}},
          importance_ops={'in_path': '22_04_results/unfiltered/importance_data.xlsx', 'name': 'H'},
          qheatmap_ops=('.', ))
exit(0)

x = get_filtered_frame(x, delete_names=True, delete_exps=False, good_names=good_descrs, good_exps=good_exp_inds)
x.reset_index(drop=True, inplace=True)

arts, inds = np.unique(x['PaperID'], return_index=True)
inds = x.index[inds]
inds = np.sort(inds)
articles_names = x.loc[inds, 'PaperID']
print(articles_names.to_numpy())

flag_encoder = True

FILL_VALUE = -2

# mode = 'ratio'
# flag_encoder = False
# FILL_VALUE = 1
# if mode == 'difference':
#     FILL_VALUE = 0
# x = get_frame_of_ratios(x, articles_names, mode=mode)

# find_large_intersect_nominal_descr(x, clean_list(x, nominal_descr, inplace=False))
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

delete_empty_descriptors(x, descrs)

# !!! не забудь раскомменировать строчку 564
# score_mod = 'relative'
# article_analysis(x, articles_names, remove_many(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), out_file_name='Check_analysis.xlsx', predict_file_name='Useless.xlsx', predict_to_file=False, model_name='RidgeCV', score_mod=score_mod)
# get_articles_picture('Check_analysis.xlsx', 'check_picture.png', score_mod=score_mod)
# x.to_excel('Check_x.xlsx')
# exit(0)

count_sparsity_plotting_bar(x.loc[:, remove_many(x.columns, ['PaperID', 'Bad'])], create_bar=True,  # fig 2, S2
                            out_file_path=f'{PLOT_FOLDER}/fig2{EXT}')

# descr_sparcity_table(x, x.columns, all_table=True, out_folder=PLOT_FOLDER)    # fig1

# x.to_excel('x_file.xlsx')

# all_plots('new_res_27_12_21/')   # the work is not finished on this!
exit(0)

# for name in x.columns:
#     get_descr_distribution_picture(x, name, out_folder='22_04_results/descrs_distribution/')
# exit(0)

# for d in exp_descr:
#     descr_correlate_picture(x, d, 'H', out_folder='Correlations/')
# exit(0)

clean_list(x, str_descr)
if flag_encoder:
    use_label_encoder(x, str_descr)

get_normal(x, norm_nominal=True)

# print(dict_labels)

# qheatmap_data(x, descr='H', features=np.array(best_features),
#               folder='NoFilterTry_22_04_14/',
#               model='ExtraTR', recovering='const')
quality_heatmap(out_folder=PLOT_FOLDER)    # fig5

# x.loc[:, clean_list(x, nominal_descr, inplace=False)].to_excel('x_file_3.xlsx')

# exit(0)

# get_scatter_plots(recovery_data(x, 'H', recovery_method='const', fill_value=FILL_VALUE), 'H', ['SubType', 'ChambPress', 'CathDist', 'ResidPress'], out_folder='Scatters/')
# exit(0)

# x['SubType'][x['SubType'].isna()] = -1
# meanH_for_subType = x.groupby('SubType')['H'].mean()
# rH = [x.loc[i, 'H']/meanH_for_subType[x.loc[i, 'SubType']] for i in range(x.shape[0])]
# x['rH'] = rH
# mech_descr.append('rH')
# dict_ind['rH'] = copy.deepcopy(dict_ind['H'])

# get_quality_1_2_3_4(x, exp_descr, 'H', model_name='ExtraTR', flags=(1, 1, 1, 1, 0), cv_parts=4, out_folder='NoFilterTry_22_04_14/', recovery_method='const', fill_value=FILL_VALUE)

# current_cols = remove_many(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias'])
# descr_analysis(x, current_cols)
# bars_for_descr_analysis()

# data_analysis(x)
# exit(0)

score_mode = 'relative'

# !!!recovering
# recove_x = recovery_data(x, remove_many(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), remove_many(x.columns, ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), recovery_method='iterative', num_iter=500)
# recove_x.to_excel('ResultRecovering.xlsx')

# article_analysis(x, articles_names, remove_many(list(x.columns), ['PaperID', 'Bad', 'CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']), predict_to_file=False, model_name='ExtraTR', score_mod=score_mode,
#       out_folder='22_04_results')
# get_articles_picture('22_04_results/', score_mod=score_mode)

# x = recovery_data(x, exp_descr, recovery_method='const', fill_value=FILL_VALUE)
# get_result(x, ['H'], exp_descr, fill_value=FILL_VALUE, crossval_mode='3:1', count_importance=True, folder='22_04_results/unfiltered/', articles=articles_names, draw_picture=True)
# get_result(x, mech_descr, exp_descr, fill_value=FILL_VALUE, crossval_mode='3:1', folder='InverseProblem_12_01_22/', articles=articles_names, draw_picture=False)

# %INVERSE PROBLEM
# get_result(x, remove_many(exp_descr, nominal_descr), ['H'], fill_value=FILL_VALUE, crossval_mode='3:1', count_importance=False, folder='InverseProblem_12_01_22/', articles=articles_names, draw_picture=False)
# get_result_one_target_per_all_others(x, remove_many(exp_descr, nominal_descr), ['H'], fill_value=FILL_VALUE, crossval_mode='3:1', count_importance=False, folder='InverseProblem_12_01_22/', articles=articles_names, draw_picture=False)

# bar_descrs = ['H', 'E', 'CoatMu', 'CritLoad']
bar_descrs = ['H']
# add_text={'s': 'filtered frame', 'x': 0.0, 'y': 0.9}
bar_for_get_result('22_04_results/unfiltered/ModelingResults.xlsx',    # fig 3
                   out_folder=PLOT_FOLDER,
                   one_more_file_path='22_04_results/filtered/ModelingResults.xlsx',
                   out_file_name='fig3',
                   bar_descrs=clean_list(x, bar_descrs, inplace=False),
                   add_text_plot=[(0.56, 0.95, 'ExtraTrees'), (0.72, 0.95, 'SVM'), (0.8, 0.95, 'RidgeCV'), ],
                   text_plot_ops={'transform': True},
                   )
importance_bars('22_04_results/unfiltered/importance_data.xlsx', '221010_pictures', 'H')   # fig4

# model_create_frame_fit(recovery_data(x, exp_descr, recovery_method='knn', fill_value=FILL_VALUE), mech_descr, exp_descr, create_method='ExtraTR', crossval_mode='3:1', mod='all', draw_picture=False, out_folder_picture='Scatters/Scatters_predict/table3/')

# descriptor.getAnalyticFormulasForGivenFeatures(recovery_data(x, exp_descr, recovery_method='const', fill_value=FILL_VALUE).loc[dict_ind['H'], :], exp_descr, 'H', output_file='formulas.txt')

exit(0)

# x.sum().to_csv(f'{folder_prefix}debug{postfix}.csv')
# x.to_excel(f'{folder_prefix}ResultRecovering{postfix}.xlsx')
# model_create_frame_fit(x, mech_descr, exp_descr, create_method='ExtraTR', file_name=f'{folder_prefix}ResultModeling{postfix}.xlsx')
