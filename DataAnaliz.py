import pandas as pd
import numpy as np
#import sklearn as skl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import RidgeCV


def add_list(list1, list2) :
    output = list1.copy()
    for elem in list2 :
        if elem not in list1 :
            output.append(elem)
    return output


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
            del exp_descr[name]
        elif name in struct_descr:
            del struct_descr[name]
        elif name in mech_descr:
            del mech_descr[name]
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
            exp_descr[new_name] = matr[:, j]


def create_model(dataframe, target_columns, used_columns, create_method='RidgeCV'):
    if not isinstance(target_columns, list):
        target_columns = list(target_columns)
    if not isinstance(used_columns, list):
        used_columns = list(used_columns)
    if create_method == 'RidgeCV':
        model = RidgeCV()
        model.fit(dataframe[used_columns], dataframe[target_columns])
        output = dataframe[used_columns] + pd.DataFrame(data=model.predict(dataframe[used_columns]), columns=target_columns)
        output.to_excel('ResultModeling.xlsx')

exp_descr = ['CathNum', 'CathType', 'ArcCurr', 'DenArcCurr', 'ArcVolt', 'CoilCurr', 'ChambPress', 'N2ChambPress','N2ArRelation', 'DeposRate', 'VoltShift', 'SubT', 'MagnetSep', 'CathDist', 'AngleDepos', 'TargetComp','TargetGeom', 'MagnetCurr', 'MagnetVolt(MagnetPow)', 'MagnetMat', 'BlockParam'] + ['SubType', 'SubComp', 'SubChar', 'SubStruct', 'SubRough', 'IonClean', 'IonImplant', 'VoltIonImplant', 'TimeIonImplant', 'SubSurfStress', 'SubH', 'SubE', 'SubHE', 'SubH3E2', 'SubMu', 'SubJ', 'SubJk'] + ['Hum']

struct_descr = ['DenCoat', 'CoatComp', 'CoatCompPercent', 'CoatPhaseComp', 'CoatThick', 'SubLayer', 'StructType', 'LayerThick', 'PhaseDisp', 'CoatIntDefect', 'LatStrain', 'CoatSurfDefect', 'CoatIntStress', 'CoatRough']

mech_descr = ['H', 'E', 'HE', 'H3E2', 'CoatMu', 'Lc1', 'Lc2', 'CoatJ', 'CoatJk', 'EroDurab']


names = pd.read_excel('DataTable.xlsx', usecols='C', skiprows=1)['ShortName'].tolist()

x = pd.read_excel('DataTable.xlsx', usecols='E: CC', skiprows=1)
x = x.T
x.reset_index(drop=True, inplace=True)

x.columns = names

for name in names :
    ind = pd.notnull(x[name])
    if len(np.unique(x.loc[ind, name])) <= 1 :
        del x[name]
        if name in exp_descr :
            exp_descr.remove(name)
        elif name in struct_descr :
            struct_descr.remove(name)
        elif name in mech_descr :
            mech_descr.remove(name)

# print(x)

exp_descr = x.loc[:, exp_descr]
struct_descr = x.loc[:, struct_descr]
mech_descr = x.loc[:, mech_descr]

# descr = x.columns
# x = recovery_data(x, list(x.columns), recovery_method='iterative')
# print(x)

# x = pd.DataFrame(data=x, columns=list(descr))

# enc = OneHotEncoder()
#
# for name in ['SubType'] :
#     matr = enc.fit_transform(np.array(x[name]).reshape(-1, 1))
#     matr = matr.toarray()
#     del x[name]
#     if name in exp_descr :
#         del exp_descr[name]
#     elif name in struct_descr :
#         del struct_descr[name]
#     elif name in mech_descr :
#         del mech_descr[name]
#     value_names = enc.categories_[0]
#     for j in range(len(value_names) - 1) :
#         new_name = name + '_' + value_names[j]
#         x[new_name] = matr[:, j]
#         exp_descr[new_name] = matr[:, j]

use_one_hot_encoder(x, ['SubType'])

descr = x.columns
# x = recovery_data(x, list(x.columns), recovery_method='iterative')
recovery_data(x, exp_descr, recovery_method='iterative')
recovery_data(x, struct_descr, (exp_descr + struct_descr).columns, recovery_method='iterative')
recovery_data(x, mech_descr.columns, (exp_descr + struct_descr).columns, recovery_method='iterative')

create_model(x, struct_descr, exp_descr)

x.to_excel('ResultRecovering.xlsx')