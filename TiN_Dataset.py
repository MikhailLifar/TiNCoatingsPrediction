from TiN_utils import *

from TiN_frame_process import exp_descr, struct_descr, mech_descr, str_descr, nominal_descr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


class TiN_Dataset:

    def __init__(self, path_load_from, samples_number, columns_to_read: dict, skiprows=1):
        """

        :param path_load_from:
        :param samples_number:
        :param columns_to_read: {'first', 'last', 'descrs', 'rating'}. 'Last' is required, others optional
        :param skiprows:
        """
        for k, v in {'first': 'F', 'descrs': 'C', 'rating': 'E'}.items():
            if k not in columns_to_read:
                columns_to_read[k] = v
        self.data_path = path_load_from
        self.df = pd.read_excel(path_load_from, usecols=f'{columns_to_read["first"]}:{columns_to_read["last"]}',
                                skiprows=skiprows)
        assert self.df.shape[1] == samples_number
        self.descr_names = pd.read_excel(path_load_from, usecols=columns_to_read["descrs"], skiprows=skiprows).to_numpy()[:, 0]
        self.df = self.df.T
        self.df.reset_index(drop=True, inplace=True)
        self.df.columns = self.descr_names
        self.descr_rating = pd.read_excel(path_load_from, usecols=columns_to_read["rating"], skiprows=skiprows).to_numpy()[:, 0]
        self.good_samples = np.arange(self.df.shape[0])[self.df['Bad'].isna()]

        # self.delete_empty_invalid_names()

        # self.filtered = False

        # self.descrs_sparsity = self.samples_sparcity = None

        self.dict_labels = self.dict_norm = self.dict_idxs = None

    def delete_empty_invalid_names(self):
        invalid_descrs = np.zeros_like(self.descr_names, dtype='bool')
        descr_names_copy = copy.deepcopy(self.descr_names)

        for i, name in enumerate(self.descr_names):
            ind = pd.notnull(self.df[name])
            if (len(np.unique(self.df.loc[ind, name])) <= 1) or (name[:5] == 'Addit'):
                del self.df[name]
                invalid_descrs[i] = 1

        self.descr_names = self.descr_names[~invalid_descrs]
        self.descr_rating = self.descr_rating[~invalid_descrs]
        invalid_names = descr_names_copy[invalid_descrs]

        print(f'Following descriptors were deleted: {invalid_names}')
        print(f'Frame shape after deleting empty: {self.df.shape}')

    def apply_filter(self, threshold_for_descrs=None, filter_samples=False):
        if threshold_for_descrs is not None:
            good_names = self.descr_names[self.descr_rating >= threshold_for_descrs]
            if filter_samples:
                self.df = self.df.loc[self.good_samples, good_names]
            else:
                self.df = self.df.loc[:, good_names]
            self.descr_names = self.descr_names[self.descr_rating >= threshold_for_descrs]
            self.descr_rating = self.descr_rating[self.descr_rating >= threshold_for_descrs]
        elif filter_samples:
            self.df = self.df.loc[self.good_samples, :]
        else:
            return

        self.df.reset_index(drop=True, inplace=True)

    def apply_filter_descr_values_ranges(self, samples_minimum=100, **name_values_pairs):
        idxs = np.ones(self.df.shape[0])
        for name, v in name_values_pairs.items():
            if isinstance(v, str) or isinstance(v, int):
                idxs = idxs & (self.df[name] == v)
            elif isinstance(v, tuple):
                idxs = idxs & (self.df[name] >= v[0]) & (self.df[name] <= v[1])
        assert np.sum(idxs) >= samples_minimum

        if np.all(idxs):
            return

        self.df = self.df.loc[idxs, :]
        self.df.reset_index(drop=True, inplace=True)

    def get_articles_names(self):
        articles, articles_inds = np.unique(self.df['PaperID'], return_index=True)
        articles_inds = np.sort(self.df.index[articles_inds])
        articles_names = self.df.loc[articles_inds, 'PaperID']
        # print(articles_names.to_numpy())
        return articles_names

    def remember_not_missed_for_each_descr(self):
        self.dict_idxs = dict()  # reinitializing
        for name in self.descr_names:
            idxs = pd.notnull(self.df[name])
            if (len(np.unique(self.df.loc[idxs, name])) > 1) and (name[:5] != 'Addit'):
                self.dict_idxs[name] = self.df.index[idxs].to_numpy()

    def normalize(self, norm_nominal=False):
        data_arr = self.df.to_numpy()
        columns = self.descr_names
        self.dict_norm = dict()
        for j, col in enumerate(columns):
            if (col not in nominal_descr) or norm_nominal:
                try:
                    notna_arr = data_arr[:, j].astype('float64')
                    notna_arr = np.concatenate((notna_arr[notna_arr < 0], notna_arr[notna_arr >= 0]))
                    data_arr[:, j] -= np.mean(notna_arr)
                    data_arr[:, j] /= np.std(notna_arr)
                    self.dict_norm[columns[j]] = [np.mean(notna_arr), np.std(notna_arr)]
                except ValueError:
                    print(f'Unable to normalize col {col}: {col} is not numeric!')
            else:
                self.dict_norm[col] = [1, 1]

    def apply_OneHotEncoder(self, names, nan_processor='ignore'):
        enc = OneHotEncoder()

        for name in names:
            arr = enc.fit_transform(np.array(self.df[name]).reshape(-1, 1))
            arr = arr.toarray()
            del self.df[name]
            for li in (exp_descr, struct_descr, mech_descr):
                if name in li:
                    li.remove(name)
            value_names = enc.categories_[0]
            length = len(value_names)
            if isinstance(value_names[-1], float):
                if nan_processor == 'ignore':
                    length = len(value_names) - 1
                elif nan_processor == 'add':
                    value_names[-1] = 'NaN'
            for j in range(length):
                new_name = name + '_' + value_names[j]
                self.df[new_name] = arr[:, j]
                self.dict_idxs[new_name] = self.dict_idxs[name]
                exp_descr.append(new_name)
            del self.dict_idxs[name]

    def apply_LabelEncoder(self, names):
        self.dict_labels = dict()

        for name in names:
            enc = LabelEncoder()
            source_col = self.df[name]
            new_col = enc.fit_transform(source_col[source_col.notna()].to_numpy())
            self.df[name][source_col.notna()] = new_col
            self.dict_labels[name] = enc.classes_

    def get_descrs_samples_sparsity(self, dest_filepath_descrs=None):
        # TODO change functionality outside the class
        missing_mask = self.df.isna().to_numpy()
        missing_counts_descrs = np.sum(missing_mask, axis=0)
        missing_counts_samples = np.sum(missing_mask, axis=1)
        if dest_filepath_descrs is not None:
            with open(dest_filepath_descrs, 'w') as f:
                f.write(f'file with data path: {self.data_path}\n')
                f.write('-----SparsityData-----\n')
                f.write('descriptor;sparsity\n')
                to_file_col = missing_counts_descrs / self.df.shape[0]
                for i, s in enumerate(to_file_col):
                    f.write(f'{self.descr_names[i]};{s}\n')
        return missing_counts_descrs, missing_counts_samples
