#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 下午8:59
# @Author  : Jun
# @File    : test.py

import json
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


def read_data_csv(file_path, key):
    table_names = read_table_names(file_path)
    for idx, pth in enumerate(table_names):
        print(idx, pth)
        with open(pth) as f:
            bom = json.load(f)
            if key in bom and len(bom[key])>0:
                idx += 1
                print(len(bom[key]))
                print(bom[key])
                # print(key + '  in the data')
            else:
                os.remove(pth)


if __name__ == '__main__':
    pth = '/Users/jun/home/data/json'
    tmp = read_data_csv(pth, 'MX_JD_SUMMARY|PULL')
