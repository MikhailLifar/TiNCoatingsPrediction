import copy
import numpy as np
import pandas as pd


def clean_list_of_names(df: pd.DataFrame, names: list, inplace=True):
    """
    
    The function deletes from list of names
    each name that is not name of dataframe column
    
    :param df: 
    :param names: 
    :param inplace: 
    :return: 
    """

    columns = df.columns
    i = 0
    if inplace:
        assert isinstance(names, list)
        while i < len(names):
            if names[i] not in columns:
                names.pop(i)
            else:
                i += 1
    else:
        output = []
        while i < len(names):
            if names[i] in columns:
                output.append(names[i])
            i += 1

        return output


def del_per_index(li, i):
    return li[:i] + li[i + 1:]


def remove_many(li, del_li):
    """
    The function returns list of elements from 1st object
    which are not elements of second object

    :param li:
    :param del_li:
    :return: list
    """

    output = []
    for elem in li:
        if elem not in del_li:
            output.append(elem)

    return output


def lists_union(li1, li2):
    """
    The function returns the list that is the union of objects,
    order of elements isn't changing
    :param li1:
    :param li2:
    :return:
    """
    output = copy.deepcopy(li1)
    for elem in li2:
        if elem not in li1:
            output.append(elem)
    return output


def lists_intersection(li1, li2):
    out = []
    for elem in li1:
        if elem in li2:
            out.append(elem)
    return out


def color_arr_from_arr(arr, init_color=(0.6, 0, 0.9), finish_color=(0.9, 0.9, 0.5), bottom=0., up=0.):
    """

    :param arr: 1-D array od scalars
    :param init_color:
    :param finish_color:
    :param bottom:
    :param up:
    :return: 2-D array, each row is RGB value
    """
    assert isinstance(arr, np.ndarray), 'arr should be np.ndarray'
    if not (bottom or up):
        bottom = np.min(arr)
        up = np.max(arr)
    else:
        bottom = np.min([bottom, np.min(arr)])
        up = np.max([up, np.max(arr)])
    new_arr = (arr - bottom)/(up - bottom)
    assert (np.max(new_arr) <= 1 + 1e-5) and (np.min(new_arr) >= - 1e-5), 'Error!'
    init_color = np.array(init_color)
    finish_color = np.array(finish_color)
    difference = finish_color - init_color
    out_arr = np.zeros((new_arr.size, 3))
    for j in range(3):
        out_arr[:, j] = init_color[j] + difference[j] * new_arr
    return out_arr


def get_random_values(obj, distribution='random'):
    """

    :param obj: 1-D container of numbers
    :param distribution:
    :return: 1-D array with the same size as input container
    """

    if distribution == 'same_distribution':
        if isinstance(obj, pd.core.series.Series):
            mass = obj.to_numpy()
        else:
            mass = np.array(obj)
        uniq_values = np.unique(mass)
        for val in uniq_values:
            mass[mass == val] = np.random.normal()
        np.random.shuffle(mass)
        return mass
    elif distribution == 'shuffle_origin':
        if isinstance(obj, pd.core.series.Series):
            mass = obj.to_numpy()
        else:
            mass = np.array(obj)
        np.random.shuffle(mass)
        return mass
    elif distribution == 'random':
        return np.random.normal(size=len(obj))
    else:
        assert False, 'invalid value for distribution'


def scoreFast(y, predictY):
    if len(y.shape) == 1:
        u = np.mean((y - predictY)**2)
        v = np.mean((y - np.mean(y))**2)
    else:
        u = np.mean(np.linalg.norm(y - predictY, axis=1, ord=2)**2)
        v = np.mean(np.linalg.norm(y - np.mean(y, axis=0).reshape([1,y.shape[1]]), axis=1, ord=2)**2)
    if v == 0:
        return 0.
    return 1. - u / v


def scoreByConstPredict(y, predictY, predictConst):
    assert len(y.shape) == 1
    u = np.mean((y - predictY)**2)
    v = np.mean((y - predictConst)**2)
    return 1-u/v


def delete_row(dataframe, ind):
    return dataframe.drop(dataframe.index[ind])


def get_random_group(num_cols, len_col, type_dependence='no', missing_chance=[], papers=None):
    assert len(papers) == len_col
    if not len(missing_chance):
        missing_chance = np.full(num_cols, 0.5)
    if type_dependence == 'no':
        mass_val = np.random.random((len_col, num_cols))
    elif type_dependence == 'proportion':
        assert num_cols == 2, 'num_cols is should be 2 for proportion'
        mass_val = np.random.random((len_col, 2))
        mass_val[:, 1] = 2 * mass_val[:, 0]
    elif type_dependence == 'multiply':
        assert num_cols == 3, 'num_cols is should be 3 for multiply'
        mass_val = np.random.random((len_col, 3))
        mass_val[:, 2] = mass_val[:, 0] * mass_val[:, 1]
    elif type_dependence == 'summ':
        mass_val = np.random.random((len_col, num_cols))
        mass_val[:, num_cols - 1] = np.sum(mass_val[:, :num_cols - 1], axis=1)
    else:
        assert False, 'invalid value for type_dependence'
    uniqPapers = np.unique(papers)
    for j in range(num_cols):
        missing_mask = np.random.random(len(uniqPapers)) < missing_chance[j]
        nanPapers = uniqPapers[missing_mask]
        for p in nanPapers:
            mass_val[papers == p, j] = np.nan
    badPapers = uniqPapers[:5]
    if num_cols == 10:
        print('Bad papers: ', badPapers)
    for p in badPapers:
        ind = np.where(papers == p)[0]
        for i in ind:
            for j in range(mass_val.shape[1]):
                if not np.isnan(mass_val[i, j]):
                    mass_val[i, j] = np.random.random()
    return mass_val

