#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 14:55
# @Author  : Jun
# @File    : test_code.py

from py_code import Qread
from py_code import Qexplore
import os
import numpy as np
import pandas as pd
import argparse


def par():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt_pth', help="the base data")
    parser.add_argument('--dt_lbl', help="the base label")
    parser.add_argument('--dt_opt', help="the output")
    args = parser.parse_args()
    return args


def read_data(dt_pth, data_key):
    print('-----------read data------------------')
    df = Qread.read_data_csv(dt_pth, data_key)
    df.to_csv(dt_output + '/dataset.csv', index=False)
    return df.shape


def preprocessing_data(dt_output, data_key, exclude_vars=[]):
    print('-----------start------------------')
    df = Qread.read_csv(dt_output + '/dataset.csv')
    print(df.shape)
    print('----------------explore the data-------------------')
    data_explore = Qexplore.data_explore(df, dt_output=dt_output, exclude_vars=[data_key])
    return data_explore


if __name__ == "__main__":
    # args = par()
    # dt_pth = args.dt_pth
    # dt_output = args.dt_opt
    dt_output = '/Users/jun/home/data/logistical_regression/output'
    dt_pth = '/Users/jun/home/data/logistical_regression/data'
    data_key = 'apply_no'
    print(read_data(dt_pth, data_key))
    print(preprocessing_data(dt_output, data_key))
    print('----------the run is done---------')
    


