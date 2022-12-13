from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from TiN_utils import *


def filter_df_per_descr_values_ranges(df, samples_minimum=100, **name_values_pairs):
    idxs = np.ones(df.shape[1])
    for name in name_values_pairs:
        for v in name_values_pairs[name]:
            if isinstance(v, str) or isinstance(v, int):
                idxs = idxs & (df[name] == v)
            elif isinstance(v, tuple):
                idxs = idxs & (df[name] >= v[0]) & (df[name] <= v[1])
    assert np.sum(idxs) >= samples_minimum

    return idxs


def find_large_intersect_nominal_descr(df, descrs):
    li = []
    for i in range(df.shape[0]):
        elem = ''
        for descr in descrs:
            elem += str(df.loc[i, descr]) + ' '
        li.append(elem)
    vals, counts = np.unique(li, return_counts=True)
    print(vals)
    print(counts)


def recover_dataframe(df, names_to_recover, additional_names=None, recovery_method='simple', fill_value=0, num_iter=50):
    if recovery_method == 'hand_made':
        return df

    if not isinstance(names_to_recover, list):
        names_to_recover = list(names_to_recover)

    if additional_names is None:
        additional_names = []
    if not isinstance(additional_names, list):
        additional_names = list(additional_names)

    fit_columns = lists_union(names_to_recover, additional_names)
    output = copy.deepcopy(df)
    if recovery_method == 'simple':
        imp = SimpleImputer(strategy='most_frequent')
    elif recovery_method == 'iterative':
        imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=num_iter, initial_strategy='most_frequent', random_state=0)
        # imp_classifier = IterativeImputer(estimator=ExtraTreesClassifier(), max_iter=num_iter, initial_strategy='most_frequent', random_state=0)
    elif recovery_method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    elif recovery_method == 'const':
        imp = SimpleImputer(strategy='constant', fill_value=fill_value)
        # imp_classifier = SimpleImputer(strategy='constant', fill_value=str(fill_value))

    # elif recovery_method == 'soft_imp':
    #     imp = SoftImpute()
    # elif recovery_method == 'nnm':
    #     imp = NuclearNormMinimization(verbose=True)
    # elif recovery_method == 'bi_scaler':
    #     imp = BiScaler()
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
        matr = imp.fit_transform(df[fit_columns])

    output.loc[:, names_to_recover] = matr[:, :len(names_to_recover)]

    return output


# def recove_and_normalize(frame, target_columns, used_columns=None, recovery_method='simple', fill_value=0, num_iter=50, **key_args):
#     new_frame = recover_dataframe(frame, target_columns, used_columns, recovery_method, fill_value, num_iter)
#     normalize_frame(new_frame, norm_nominal=key_args['norm_nominal'])
#     return new_frame


def generate_df_with_diff_dependencies(frame, main_chance=0.3):
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


# TODO try to delete, then check quality
del_names = ['DenArcCurr', 'CathNum', 'CathType']

exp_descr = ['Method'] + ['CathNum', 'CathType', 'ArcCurr', 'DenArcCurr', 'ArcVolt', 'CoilCurr', 'ChambPress',
                          'N2ChambPress','N2ArRelation', 'DeposTime', 'VoltBias', 'SubT', 'MagnetSep', 'CathDist',
                          'AngleDepos', 'TargetComp','TargetGeom', 'TargetSize', 'MagnetCurr', 'MagnetVolt(MagnetPow)',
                          'MagnetMat', 'BlockParam'] + ['SubType', 'SubComp', 'SubChar', 'SubStruct', 'SubRough',
                                                        'IonClean', 'IonImplant', 'VoltIonImplant', 'TimeIonImplant',
                                                        'SubSurfStress', 'SubH', 'SubE', 'SubHE', 'SubH3E2', 'SubMu',
                                                        'SubJ', 'SubJk'] + ['Hum', 'IndentMethod'] + ['SubLayer']\
            + ['DeposRate'] + ['TotalFlow', 'N2Flow', 'TargetPow'] + ['Method_1', 'Method_2', 'MagType', 'ResidPress',
                                                                      'PowDensity', 'IonAtomRatio'] + ['Indent']\
            + ['FricLoad', 'FricSpeed', 'BallSize', 'BallComp'] + ['React_or_not', 'Balanc_or_not']

struct_descr = ['DenCoat', 'CoatComp', 'CoatCompPercent', 'CoatPhaseComp', 'CoatThick', 'StructType', 'LayerThick',
                'PhaseDisp', 'CoatIntDefect', 'LatStrain', 'CoatSurfDefect', 'CoatIntStress'] + ['CoatRough',
                                                                                                 'GrainSize', 'Orient']

mech_descr = ['H', 'E', 'HE', 'H3E2', 'CoatMu', 'Lc1', 'Lc2', 'CoatJ', 'CoatJk', 'EroDurab'] \
             + ['CoatMu_1', 'CritLoad', 'Wear']

str_descr = ['Method', 'TargetComp', 'TargetGeom', 'SubType', 'IonImplant', 'SubLayer', 'IndentMethod', 'BallComp']
nominal_descr = ['Method_1', 'Method_2', 'React_or_not', 'Balanc_or_not', 'MagType', 'Orient'] + str_descr

check_columns = ['CheckSum_N2Press_SubT', 'CheckSum_DeposRate_VoltBias']

best_features = ['PowDensity', 'CathDist', 'ResidPress', 'SubT', 'ChambPress', 'VoltBias', 'N2ArRelation',
                 'Balanc_or_not', 'SubType']
