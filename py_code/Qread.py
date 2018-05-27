#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 14:34
# @Author  : Jun
# @File    : Qread.py

import os
import pandas as pd


def read_table_names(file_path):
    table_names = []
    if os.path.exists(file_path):
        file_names = os.listdir(file_path)
        for file_i in file_names:
            path_file = os.path.join(file_path, file_i)
            table_names.append(path_file)
        if len(table_names) == 0:
            print('there is no file in the path')
    else:
        print('the path [{}] is not exist!'.format(file_path))
    return table_names


def adjust_col(df):
    df.columns = df.columns.map(lambda x: x.lower())
    return df


def read_csv(file, index_col=None):
    df = pd.read_csv(file, low_memory=True, index_col=index_col, delimiter=',')
    return df


def read_data_csv(file_path, key_by, index_col=None):
    table_names = read_table_names(file_path)
    df = pd.DataFrame()
    for idx, pth in enumerate(table_names):
        print(pth)
        df_ = pd.DataFrame()
        if idx == 0:
            df = read_csv(pth, index_col=index_col)
            df = adjust_col(df)
        elif idx > 0:
            df_ = read_csv(pth, index_col=index_col)
            df_ = adjust_col(df_)
            df = pd.merge(df, df_, left_on=key_by, right_on=key_by, how='inner')
    return df





