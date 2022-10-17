import copy
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from usable_functions_1 import *


def get_filtered_frame(frame, delete_names=True, delete_exps=True, good_names=[], good_exps=[]):
    if delete_names:
        if delete_exps:
            return frame.loc[good_exps, good_names]
        return frame.loc[:, good_names]
    elif delete_exps:
        return frame.loc[good_exps, :]
    return frame


def delete_empty_descriptors(frame, names):
    names = clean_list(frame, names, inplace=False)

    for name in names:
        ind = pd.notnull(frame[name])
        if name[:5] != 'Addit':
            if len(np.unique(frame.loc[ind, name])) <= 1:
                del frame[name]
            else:
                dict_ind[name] = list(frame.index[ind])
        else:
            del frame[name]

    print(frame.shape)

    clean_list(frame, exp_descr)
    clean_list(frame, struct_descr)
    clean_list(frame, mech_descr)


def find_large_intersect_nominal_descr(frame, descrs):
    li = []
    for i in range(frame.shape[0]):
        elem = ''
        for descr in descrs:
            elem += str(frame.loc[i, descr]) + ' '
        li.append(elem)
    vals, counts = np.unique(li, return_counts=True)
    print(vals)
    print(counts)


def get_normal(df: pd.DataFrame, norm_nominal=False):
    """

    The function normalises the dataframe

    :param df:
    :param norm_nominal:
    :return:
    """
    # print(dataframe.dtypes)
    data_arr = df.to_numpy()
    columns = df.columns
    for j in range(data_arr.shape[1]):
        if (columns[j] not in nominal_descr) or norm_nominal:
            try:
                row_arr = data_arr[:, j].astype('float64')
                row_arr = np.concatenate((row_arr[row_arr < 0], row_arr[row_arr >= 0]))
                data_arr[:, j] -= np.mean(row_arr)
                data_arr[:, j] /= np.std(row_arr)
                dict_norm[columns[j]] = [np.mean(row_arr), np.std(row_arr)]
            except ValueError:
                print(columns[j] + ' is not numeric!')
        else:
            dict_norm[columns[j]] = [1, 1]
    for j in range(data_arr.shape[1]):
        try:
            df[columns[j]] = data_arr[:, j].astype('float64')
        except ValueError:
            print(columns[j] + ' is not continous!')
    # print(dataframe.dtypes)


def recovery_data(dataframe, target_columns, used_columns=[], recovery_method='simple', fill_value=0, num_iter=50):
    if recovery_method == 'hand_made':
        return dataframe
    if not isinstance(target_columns, list):
        target_columns = list(target_columns)
    if not isinstance(used_columns, list):
        used_columns = list(used_columns)
    fit_columns = add_list(target_columns, used_columns)
    output = copy.deepcopy(dataframe)
    if recovery_method == 'simple':
        imp = SimpleImputer(strategy='most_frequent')
    elif recovery_method == 'iterative':
        imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=num_iter, initial_strategy='most_frequent', random_state=0)
        imp_classifier = IterativeImputer(estimator=ExtraTreesClassifier(), max_iter=num_iter, initial_strategy='most_frequent', random_state=0)
    elif recovery_method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    # elif recovery_method == 'soft_imp':
    #     imp = SoftImpute()
    # elif recovery_method == 'nnm':
    #     imp = NuclearNormMinimization(verbose=True)
    # elif recovery_method == 'bi_scaler':
    #     imp = BiScaler()
    elif recovery_method == 'const':
        imp = SimpleImputer(strategy='constant', fill_value=fill_value)
        imp_classifier = SimpleImputer(strategy='constant', fill_value=str(fill_value))
    # elif recovery_method == 'matr_factoriz':
    #     imp = MatrixFactorization()
    # elif recovery_method == 'similarity':
    #     imp = SimilarityWeightedAveraging()
    # elif recovery_method == 'iter_svd':
    #     imp = IterativeSVD()
    # elif recovery_method == 'solver':
    #     imp = Solver()
    # elif recovery_method == 'datawig':
    #     imp = DwImputer(input_columns=fit_columns, output_column=target_columns)
    else:
        assert False, f'invalid value for recovery_method: {recovery_method}'
    if recovery_method != 'datawig':
        matr = imp.fit_transform(dataframe[fit_columns])
    # print(dataframe.dtypes)
    # print(output.dtypes)
    output.loc[:, target_columns] = matr[:, :len(target_columns)]
    # for j in range(len(target_columns)):
    #     output.loc[:, target_columns[j]] = matr[:, j]
    # print(output.dtypes)
    # exit(0)
    return output


def recove_and_normalize(frame, target_columns, used_columns=[], recovery_method='simple', fill_value=0, num_iter=50, **key_args):
    new_fame = recovery_data(frame, target_columns, used_columns, recovery_method, fill_value, num_iter)
    get_normal(new_fame, norm_nominal=key_args['norm_nominal'])
    return new_fame


def use_one_hot_encoder(dataframe, columns, nan_processor='ignore'):
    enc = OneHotEncoder()
    if not isinstance(columns, list):
        columns = list(columns)

    for name in columns:
        matr = enc.fit_transform(np.array(dataframe[name]).reshape(-1, 1))
        matr = matr.toarray()
        del dataframe[name]
        if name in exp_descr:
            exp_descr.remove(name)
        elif name in struct_descr:
            struct_descr.remove(name)
        elif name in mech_descr:
            mech_descr.remove(name)
        value_names = enc.categories_[0]
        length = len(value_names)
        if isinstance(value_names[-1], float):
            if nan_processor == 'ignore':
                length = len(value_names) - 1
            elif nan_processor == 'add':
                value_names[-1] = 'NaN'
        for j in range(length):
            new_name = name + '_' + value_names[j]
            dataframe[new_name] = matr[:, j]
            dict_ind[new_name] = dict_ind[name]
            exp_descr.append(new_name)
        del dict_ind[name]


def use_label_encoder(dataframe, columns):
    if not isinstance(columns, list):
        columns = list(columns)

    for name in columns:
        enc = LabelEncoder()
        name_col = dataframe[name]
        col = enc.fit_transform(name_col[name_col.notna()].to_numpy())
        dataframe[name][name_col.notna()] = col
        dict_labels[name] = enc.classes_


def check_code(frame, main_chance=0.3):
    np.random.seed(0)
    papers = frame['PaperID']
    for name in frame.columns:
        if name[:5] == 'Addit':
            del frame[name]
    cols = frame.columns
    num_descr = len(cols)
    print('random_cols :', cols[1:11], cols[34:48], sep='\n')
    print('proportion_cols :', cols[11:17])
    print('summ_cols: ', cols[17:28])
    print('multiply_cols: ', cols[28:34])
    frame.loc[:, cols[1:11]] = get_random_group(10, frame.shape[0], missing_chance=np.linspace(0.1, 0.5, 10), papers=papers)
    frame.loc[:, cols[11:13]] = get_random_group(2, frame.shape[0], type_dependence='proportion', missing_chance=[main_chance] * 2, papers=papers)
    frame.loc[:, cols[13:15]] = get_random_group(2, frame.shape[0], type_dependence='proportion', missing_chance=[main_chance] * 2, papers=papers)
    frame.loc[:, cols[15:17]] = get_random_group(2, frame.shape[0], type_dependence='proportion', missing_chance=[main_chance] * 2, papers=papers)
    frame.loc[:, cols[17:20]] = get_random_group(3, frame.shape[0], type_dependence='summ', missing_chance=[main_chance] * 3, papers=papers)
    frame.loc[:, cols[20:23]] = get_random_group(3, frame.shape[0], type_dependence='summ', missing_chance=[main_chance * 1.5] * 3, papers=papers)
    frame.loc[:, cols[23:28]] = get_random_group(5, frame.shape[0], type_dependence='summ', missing_chance=[main_chance * 4e-1] * 5, papers=papers)
    frame.loc[:, cols[28:31]] = get_random_group(3, frame.shape[0], type_dependence='multiply', missing_chance=[main_chance] * 3, papers=papers)
    frame.loc[:, cols[31:34]] = get_random_group(3, frame.shape[0], type_dependence='multiply', missing_chance=[main_chance] * 3, papers=papers)
    frame.loc[:, cols[34:48]] = get_random_group(14, frame.shape[0], missing_chance=np.linspace(0.1, 0.95, 14), papers=papers)
    frame.loc[:, cols[48:num_descr-3]] = get_random_group(num_descr - 3 - 48, frame.shape[0], missing_chance=np.ones(num_descr - 3 - 48) + 1, papers=papers)

    return frame


# TODO попробовать удалить, посмотреть на качество
del_names = ['DenArcCurr', 'CathNum', 'CathType']

exp_descr = ['Method'] + ['CathNum', 'CathType', 'ArcCurr', 'DenArcCurr', 'ArcVolt', 'CoilCurr', 'ChambPress',
                          'N2ChambPress','N2ArRelation', 'DeposTime', 'VoltBias', 'SubT', 'MagnetSep', 'CathDist',
                          'AngleDepos', 'TargetComp','TargetGeom', 'TargetSize', 'MagnetCurr', 'MagnetVolt(MagnetPow)',
                          'MagnetMat', 'BlockParam'] + ['SubType', 'SubComp', 'SubChar', 'SubStruct', 'SubRough',
                                                        'IonClean', 'IonImplant', 'VoltIonImplant', 'TimeIonImplant',
                                                        'SubSurfStress', 'SubH', 'SubE', 'SubHE', 'SubH3E2', 'SubMu',
                                                        'SubJ', 'SubJk'] + ['Hum', 'HMethod'] + ['SubLayer']\
            + ['DeposRate'] + ['TotalFlow', 'N2Flow', 'TargetPow'] + ['Method_1', 'Method_2', 'MagType', 'ResidPress',
                                                                      'PowDensity', 'IonAtomRatio'] + ['Indent']\
            + ['FricLoad', 'FricSpeed', 'BallSize', 'BallComp'] + ['React_or_not', 'Balanc_or_not']

struct_descr = ['DenCoat', 'CoatComp', 'CoatCompPercent', 'CoatPhaseComp', 'CoatThick', 'StructType', 'LayerThick',
                'PhaseDisp', 'CoatIntDefect', 'LatStrain', 'CoatSurfDefect', 'CoatIntStress'] + ['CoatRough',
                                                                                                 'GrainSize', 'Orient']

mech_descr = ['H', 'E', 'HE', 'H3E2', 'CoatMu', 'Lc1', 'Lc2', 'CoatJ', 'CoatJk', 'EroDurab'] \
             + ['CoatMu_1', 'CritLoad', 'Wear']

str_descr = []
str_descr = ['Method', 'TargetComp', 'TargetGeom', 'SubType', 'IonImplant', 'SubLayer', 'HMethod', 'BallComp']
nominal_descr = ['Method_1', 'Method_2', 'React_or_not', 'Balanc_or_not', 'MagType', 'Orient'] + str_descr

check_columns = ['CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']

best_features = ['PowDensity', 'CathDist', 'ResidPress', 'SubT', 'ChambPress', 'VoltBias', 'N2ArRelation',
                 'Balanc_or_not', 'SubType']

# глобальные переменные словари
# словарь, содержащий идентификатор последнего столбца с данными для разных файлов
dict_last_cols = dict()
# словарь, содержащий число экспериментов для разных файлов
dict_num_exps= dict()
# словарь, содержащий списки индексов непустых ячеек для каждого деск-ра
dict_ind = dict()
# словарь, содержащий списки параметров нормировки (mean, std) для каждого деск-ра
dict_norm = dict()
# словарь, содержащий списки значений, соотв.- их меткам для строковых деск-ров
dict_labels = dict()
