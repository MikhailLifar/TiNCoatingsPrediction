import pandas as pd
import numpy as np
#import sklearn as skl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import RidgeCV


def del_per_index(li, i) :
    return li[:i] + li[i + 1:]


def add_list(list1, list2):
    output = list1.copy()
    for elem in list2:
        if elem not in list1:
            output.append(elem)
    return output


def delete_row(dataframe, ind):
    return dataframe.drop(dataframe.index[ind])


def get_normal(dataframe):
    data_arr = dataframe.to_numpy()
    data_arr = data_arr.transpose()
    columns = list(dataframe.columns)
    for i in range(data_arr.shape[0]):
        try:
            row_arr = np.float32(data_arr[i])
            row_arr = np.concatenate((row_arr[row_arr < 0], row_arr[row_arr >= 0]))
            data_arr[i] -= np.mean(row_arr)
            data_arr[i] /= np.std(row_arr)
            dict_norm[columns[i]] = [np.mean(row_arr), np.std(row_arr)]
        except ValueError:
            print('Error!')
    for i in range(data_arr.shape[0]):
        dataframe[columns[i]] = data_arr[i]


def recovery_data(dataframe, target_columns, used_columns=[], recovery_method='simple'):
    if not isinstance(target_columns, list):
        target_columns = list(target_columns)
    if not isinstance(used_columns, list):
        used_columns = list(used_columns)
    fit_columns = add_list(target_columns, used_columns)
    if recovery_method == 'simple':
        imp = SimpleImputer(strategy='most_frequent')
        matr = imp.fit_transform(dataframe[fit_columns])
        for j in range(len(target_columns)):
            dataframe[target_columns[j]] = matr[:, j]
    elif recovery_method == 'iterative':
        imp = IterativeImputer(max_iter=200, initial_strategy='most_frequent', random_state=0)
        matr = imp.fit_transform(dataframe[fit_columns])
        for j in range(len(target_columns)):
            dataframe[target_columns[j]] = matr[:, j]


def use_one_hot_encoder(dataframe, columns, nan_processor='ignore') :
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
            exp_descr.append(new_name)


def create_model(dataframe, target_columns, used_columns, create_method='RidgeCV'):
    if not isinstance(target_columns, list):
        target_columns = list(target_columns)
    if not isinstance(used_columns, list):
        used_columns = list(used_columns)
    if create_method == 'RidgeCV':
        model = RidgeCV()
        output = pd.DataFrame()
        for name in target_columns:
            inds = dict_ind[name]
            original_values = np.array([None] * 70)
            values = np.array([None] * 70)
            for i in range(len(inds)):
                original_values[i] = dataframe.loc[inds[i], name] * dict_norm[name][1] + dict_norm[name][0]
                model.fit(dataframe.loc[del_per_index(inds, i), used_columns], dataframe.loc[del_per_index(inds, i), name])
                values[i] = model.predict(dataframe.loc[dict_ind[name], used_columns])[i] * dict_norm[name][1] + dict_norm[name][0]
            output[name + '_original'] = original_values
            output[name + '_predict'] = values
        output.to_excel('ResultModeling.xlsx')


exp_descr = ['CathNum', 'CathType', 'ArcCurr', 'DenArcCurr', 'ArcVolt', 'CoilCurr', 'ChambPress', 'N2ChambPress','N2ArRelation', 'DeposRate', 'VoltShift', 'SubT', 'MagnetSep', 'CathDist', 'AngleDepos', 'TargetComp','TargetGeom', 'MagnetCurr', 'MagnetVolt(MagnetPow)', 'MagnetMat', 'BlockParam'] + ['SubType', 'SubComp', 'SubChar', 'SubStruct', 'SubRough', 'IonClean', 'IonImplant', 'VoltIonImplant', 'TimeIonImplant', 'SubSurfStress', 'SubH', 'SubE', 'SubHE', 'SubH3E2', 'SubMu', 'SubJ', 'SubJk'] + ['Hum']

struct_descr = ['DenCoat', 'CoatComp', 'CoatCompPercent', 'CoatPhaseComp', 'CoatThick', 'SubLayer', 'StructType', 'LayerThick', 'PhaseDisp', 'CoatIntDefect', 'LatStrain', 'CoatSurfDefect', 'CoatIntStress', 'CoatRough']

mech_descr = ['H', 'E', 'HE', 'H3E2', 'CoatMu', 'Lc1', 'Lc2', 'CoatJ', 'CoatJk', 'EroDurab']


names = pd.read_excel('DataTable.xlsx', usecols='C', skiprows=1)['ShortName'].tolist()

x = pd.read_excel('DataTable.xlsx', usecols='E: CC', skiprows=1)
x = x.T
x.reset_index(drop=True, inplace=True)

x.columns = names

dict_ind = dict()

for name in names:
    ind = pd.notnull(x[name])
    if len(np.unique(x.loc[ind, name])) <= 1:
        del x[name]
        if name in exp_descr:
            exp_descr.remove(name)
        elif name in struct_descr:
            struct_descr.remove(name)
        elif name in mech_descr:
            mech_descr.remove(name)
    else:
        ind = ind.to_numpy()
        mass = []
        for i in range(len(ind)):
            if ind[i]:
                mass.append(i)
        dict_ind[name] = mass

dict_norm = dict()

get_normal(x)

use_one_hot_encoder(x, ['SubType'])

descr = x.columns
recovery_data(x, exp_descr, recovery_method='iterative')
recovery_data(x, struct_descr, (exp_descr + struct_descr), recovery_method='iterative')
recovery_data(x, mech_descr, (exp_descr + struct_descr), recovery_method='iterative')

create_model(x, struct_descr, exp_descr)

# x = delete_row(x, 3)

x.to_excel('ResultRecovering.xlsx')