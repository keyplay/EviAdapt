'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import argparse
import os
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen


seed = 0
random.seed(0)
np.random.seed(seed)

domain='DS02'
keep_ratio = 1

data_filedir = os.path.join('../data_set')
if domain == 'DS01':
    data_filepath = os.path.join(data_filedir, 'N-CMAPSS_DS01-005.h5')
elif domain == 'DS02':
    data_filepath = os.path.join(data_filedir, 'N-CMAPSS_DS02-006.h5')
elif domain == 'DS03':
    data_filepath = os.path.join(data_filedir, 'N-CMAPSS_DS03-012.h5')





def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='window length')
    parser.add_argument('-s', type=int, default=10, help='stride of window')
    parser.add_argument('--sampling', type=int, default=100, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')

    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    sampling = args.sampling



    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    df_train, df_test, unit_flight_classes = df_all_creator(data_filepath, sampling)

    '''
    Split dataframe into Train and Test
    Training units: 2, 5, 10, 16, 18, 20
    Test units: 11, 14, 15

    '''
    # units = list(np.unique(df_A['unit']))
    units_index_train = np.unique(np.array(df_train['unit']))
    units_index_test = np.unique(np.array(df_test['unit']))

    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)

    # if any(int(idx) == unit_index for idx in units_index_train):
    #     df_train = df_train_creator(df_all, units_index_train)
    #     print(df_train)
    #     print(df_train.columns)
    #     print("num of inputs: ", len(df_train.columns) )
    #     df_test = pd.DataFrame()
    #
    # else :
    #     df_test = df_test_creator(df_all, units_index_test)
    #     print(df_test)
    #     print(df_test.columns)
    #     print("num of inputs: ", len(df_test.columns))
    #     df_train = pd.DataFrame()


    df_train = df_train_creator(df_train, units_index_train, keep_ratio)
    
    #print(df_train)
    #print(df_train.columns)
    print("num of inputs: ", len(df_train.columns) )


    gc.collect()
    df_all = pd.DataFrame()
    sample_dir_path = os.path.join(data_filedir, f'Samples_whole_{domain}_{keep_ratio}')
    sample_folder = os.path.isdir(sample_dir_path)
    if not sample_folder:
        os.makedirs(sample_dir_path)
        print("created folder : ", sample_dir_path)

    cols_normalize = df_train.columns.difference(['RUL', 'unit'])
    sequence_cols = df_train.columns.difference(['RUL', 'unit'])


    for unit_index in units_index_train:
        data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                unit_index, sampling, stride =stride)
        data_class.seq_gen()

    for unit_index in units_index_test:
        data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                unit_index, sampling, stride =stride)
        data_class.seq_gen()

    #np.save(os.path.join(sample_dir_path,'unit_num_train.npy'), units_index_train)
    #np.save(os.path.join(sample_dir_path,'unit_num_test.npy'), units_index_test)



if __name__ == '__main__':
    main()
