#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 10:22
# @Author  : Jun
# @File    : woe.py

import numpy as np
import pandas as pd
from py_code.binning import *
from pandas import ExcelWriter
import pickle


def ex_inf_sum(list_x, method):
    list_y = [i for i in list_x if i not in [np.inf, -np.inf, 'inf', '-inf', np.nan, 'nan', 'null', 'NAN']]
    if method == 'SUM':
        return sum(list_y)
    if method == 'MAX':
        return max(list_y)


def woe(data_df, target, save_path, exclude_var_list=[]):
    data = data_df.copy()
    var_list = data.columns
    if len(exclude_var_list) > 0:
        exclude_var_list = [var.lower() for var in exclude_var_list]
    keep_var_list = list(set(var_list) - set(exclude_var_list))
    dic_woe = {}
    dic_df = {}
    dic_type = {}
    summary = []
    num = 0
    num_2 = 0
    cant_col = []
    count_var = 0
    writer = ExcelWriter(save_path + '/iv_sheet.xlsx')
    for var_name in keep_var_list:
        print('='*50)
        print('{0} : {1}'.format(count_var, var_name))
        count_var += 1
        try:
            df_1 = find_best_bin(data=data, subset=[0, 1], y=target, var_name=var_name, groups=5, rate=0.05)
            dic_df[var_name] = df_1
            # df_1['variable'] = str(var_name)
            dic_type[var_name] = df_1['type'].max()
            woe_value = df_1['Woe']
            bin_name = df_1[var_name]
            dic_woe[var_name] = dict(zip(list(bin_name), list(woe_value)))
            summary.append([var_name, df_1.shape[0], ex_inf_sum(df_1['IV'], 'SUM'),
                        ex_inf_sum(df_1['KS'], 'MAX')])
            df_1.to_excel(writer, 'detail', startrow=num)
            num += len(df_1) + 3
        except:
            print('VAR CANNOT BE CUTTED :' + var_name)
            cant_col.append(var_name)
    summary_df = pd.DataFrame(summary,
                              columns=['variable', 'group_num', 'iv', 'ks'])
    summary_df = summary_df.sort_values('iv', ascending=False).reset_index(drop=True)
    summary_df.to_excel(writer, 'summary', startrow=0)
    writer.save()
    f = open(save_path + '/VARIABLE_WOE.pkl', 'wb+')
    pickle.dump(dic_woe, f)
    f.close()
    f = open(save_path + '/VARIABLE_TYPE.pkl', 'wb+')
    pickle.dump(dic_type, f)
    f.close()
    return dic_woe, dic_df, dic_type
