from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

first_n = 21
last_n = 45
codes = 'ACDEFGHIKLMNPQRSTVWY'
names = ['Cytosolic', 'Mitochondrial', 'Nuclear', 'Secreted']


def load_file(filename):
    df = pd.DataFrame(columns=['id', 'seq', 'cat'])
    category = filename.split('.')[0]
    print('Loaded: ', filename)
    for seq in SeqIO.parse('data/{}'.format(filename), 'fasta'):
        if 'B' not in seq.seq and 'O' not in seq.seq and 'Z' not in seq.seq and 'U' not in seq.seq and 'X' not in seq.seq:
            df = df.append({'id': seq.id, 'seq': seq.seq,
                            'cat': category}, ignore_index=True)
    return df


def load_data():
    dfs_train = [load_file(filename) for filename in os.listdir(
        'data') if filename != 'blind.fasta']
    df_train = pd.concat(dfs_train, ignore_index=True)

    df_train_y = df_train.pop('cat')
    df_train_x = df_train

    df_test_x = load_file('blind.fasta')
    df_test_x.pop('cat')

    return df_train_x, df_train_y, df_test_x


def find_composition(df_original):
    df_copy = df_original.copy()

    column_names = []
    for ch in codes:
        column_names.append(ch + '_percent')
        column_names.append(ch + '_percent_first')
        column_names.append(ch + '_percent_last')
    column_names.append('len')
    column_names.append('weight')
    column_names.append('gravy')
    column_names.append('flex_mean')
    column_names.append('flex_std')
    column_names.append('ss_helix')
    column_names.append('ss_turn')
    column_names.append('ss_sheet')
    column_names.append('iep')
    column_names.append('aromaticity')

    df = pd.DataFrame(columns=column_names)
    for _, seq in enumerate(tqdm(df_copy['seq'])):
        df_temp = pd.Series()
        sequence = str(seq)
        analysed = ProteinAnalysis(sequence)
        analysed_first = ProteinAnalysis(sequence[:first_n])
        analysed_last = ProteinAnalysis(sequence[-last_n:])

        df_temp['len'] = analysed.length
        df_temp['ss_helix'], df_temp['ss_turn'], df_temp['ss_sheet'] = analysed.secondary_structure_fraction()
        df_temp['iep'] = analysed.isoelectric_point()

        # overall
        for aa, percent in analysed.get_amino_acids_percent().items():
            df_temp[aa + '_percent'] = percent

        # # first N
        for aa, percent in analysed_first.get_amino_acids_percent().items():
            df_temp[aa + '_percent_first'] = percent

        # last N
        for aa, percent in analysed_last.get_amino_acids_percent().items():
            df_temp[aa + '_percent_last'] = percent

        df_temp['weight'] = analysed.molecular_weight()
        df_temp['gravy'] = analysed.gravy()
        df_temp['aromaticity'] = analysed.aromaticity()
        df_temp['flex_mean'] = np.mean(analysed.flexibility())
        df_temp['flex_std'] = np.std(analysed.flexibility())
        df = df.append(df_temp, ignore_index=True)

    return pd.concat([df_copy, df], axis=1)


def standardize(df, columns):

    for column in columns:
        df[column] = (df[column] - df[column].mean()) / \
            df[column].std()  # subtract mean and divide by std

    return df


def pre_process_data_x(df):
    df1 = find_composition(df)
    df1 = standardize(df1, columns=[
                      'len', 'weight', 'gravy', 'flex_mean', 'flex_std', 'iep', 'aromaticity'])

    df1.pop('seq')
    df1.pop('id')

    return df1


def pre_process_data_y(df, dummies=False):
    if dummies:
        df1 = pd.get_dummies(df, prefix='cat', columns=['cat'])
    else:
        encoder = LabelEncoder()
        encoder.fit(df)
        df1 = pd.DataFrame(encoder.transform(df), columns=['cat'])

        for i in range(4):
            print("{} -> {}".format(encoder.inverse_transform(np.array(i)), i))
    return df1.reset_index().drop(['index'], axis=1)


def prepare_all_data(verbose=False, last=None):
    if last is not None:
        last_n = last

    df_train_x, df_train_y, df_test_x = load_data()

    if verbose:
        print(df_train_x.head())
        print(df_train_y.head())
        print(df_test_x)

    # append and process train and test set together, to have the same features
    df_all_x = df_train_x.append(df_test_x, ignore_index=True)
    df_final_all_x = pre_process_data_x(df_all_x)

    # separate the train and test set again
    df_final_train_x = df_final_all_x.iloc[:-len(df_test_x)]
    df_final_test_x = df_final_all_x.iloc[-len(df_test_x):]

    df_final_train_y = pre_process_data_y(df_train_y, dummies=False)

    # shuffle the data consistently
    order = np.random.permutation(df_train_x.index)
    df_train_x = df_final_train_x.reindex(order)
    df_train_y = df_final_train_y.reindex(order)

    if verbose:
        print(df_final_train_x.head())
        print(df_final_train_y.head())
        print(df_final_test_x.head())

    return df_final_train_x, df_final_train_y, df_final_test_x
