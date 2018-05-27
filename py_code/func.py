#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 11:14
# @Author  : Jun
# @File    : func.py

import pandas as pd
import os
from py_code import Qread


def read_path(path, sep):
    data = pd.read_csv(path, sep=sep)
    return data


def convert_lower(data):
    if isinstance(data, list):
        data = [var.lower() for var in data]
    elif isinstance(data, str):
        data.lower()
    return data


def vars_keep(importance_file, var_name, method, proportion):
    keep_vars_list = []
    if not os.path.isfile(importance_file):
        print("importance file does not exist")
        return keep_vars_list
    imp = Qread.read_csv(importance_file)
    imp.columns = [col.lower() for col in imp.columns]
    if var_name:
        var_name = var_name.lower()
    if method:
        method = method.lower()
    imp.drop_duplicates([var_name], inplace=True)
    if proportion < 1:
        keep_vars_list = list(imp.ix[imp[method] >= proportion, var_name])
    if proportion > 1:
        imp.sort_values(by=method, ascending=False, inplace=True)
        imp.index = range(len(imp))
        keep_vars_list = list(imp.ix[:proportion-1, var_name])
    if len(keep_vars_list) == 0:
        print("all the varaibles are dropped")
    return keep_vars_list


