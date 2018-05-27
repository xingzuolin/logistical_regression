#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 16:22
# @Author  : Jun
# @File    : Qexplore.py

import numpy as np
import pandas as pd


def type_num_cal(data, total_cnt):
    num_explore = []
    data_df = data.fillna(-1)
    miss_cnt = sum(data_df.map(lambda x: x == -1))
    nmiss_cnt = total_cnt - miss_cnt
    miss_pct = round(float(miss_cnt)/total_cnt, 4)
    data_nnan_df = data_df[data_df != -1]
    list_num_data = np.sort(data_nnan_df)
    num_explore.extend([total_cnt, nmiss_cnt, miss_cnt, miss_pct])
    percent = []
    try:
        for i in range(0, 101, 25):
            percent.append(np.percentile(list_num_data, i))
    except:
        percent = [-1, -1, -1, -1, -1]
    num_explore.extend(percent)
    return num_explore


def type_char_cal(data, total_cnt):
    char_explore = []
    data_df = data.fillna('_')
    miss_cnt = sum(data_df.map(lambda x: x == '_'))
    nmiss_cnt = total_cnt - miss_cnt
    miss_pct = round(float(miss_cnt) / total_cnt, 4)
    char_explore.extend([total_cnt, nmiss_cnt, miss_cnt, miss_pct])
    df_sum = data_df.value_counts()
    if miss_cnt > 0:
        df_sum = df_sum[df_sum.index != '_']
        if len(df_sum) == 0:
            char_explore.extend([0, 0, 0, 0, 0, 0])
            return char_explore
    df_sum.sort_values(ascending=True, inplace=True)
    least_value = df_sum.index[0]
    least_count = df_sum[least_value]
    least_pct = round(float(least_count)/total_cnt, 4)
    most_value = df_sum.index[-1]
    most_count = df_sum[most_value]
    most_pct = round(float(most_count)/total_cnt, 4)
    char_explore.extend([most_value, most_count, most_pct, least_value, least_count, least_pct])
    return char_explore


def data_explore(data, dt_output, exclude_vars):
    df_cols = data.columns
    total_cnt = len(data)
    exclude_vars = [var.lower() for var in exclude_vars]
    df_keep_cols = list(set(df_cols) - set(exclude_vars))
    num_explore_list = []
    char_explore_list = []
    for i, col in enumerate(df_keep_cols):
        type_len = len(set([type(k) for k in list(data[col])]))
        print(i, col, type_len)
        if type_len > 1:
            char_tmp_list = []
            char_tmp_list.append(col)
            char_data = data[col].copy()
            char_list = type_char_cal(char_data, total_cnt)
            char_tmp_list.extend(char_list)
            char_explore_list.append(char_tmp_list)
        else:
            num_tmp_list = []
            num_tmp_list.append(col)
            num_list = type_num_cal(data[col], total_cnt)
            num_tmp_list.extend(num_list)
            num_explore_list.append(num_tmp_list)
    if len(num_explore_list) > 0:
        num_explore_df = pd.DataFrame(num_explore_list)
        num_explore_df.columns = ['Variable', 'Total_cnt', 'Nmiss_cnt', 'Miss_cnt', 'Miss_pct','Q0', 'Q1', 'Q2', 'Q3', 'Q4']
        num_explore_df.to_csv(dt_output + '/NumericVariablesExploring.csv', index=False)
    if len(char_explore_list) > 0:
        char_explore_df = pd.DataFrame(char_explore_list)
        char_explore_df.columns = ['Variable', 'Total_cnt', 'Nmiss_cnt', 'Miss_cnt', 'Miss_pct', 'Most_variable','Most_cnt',\
                               'Most_pct', 'Least_variable', 'Least_cnt', 'Least_pct']
        char_explore_df.to_csv(dt_output + '/ChracterVariablesExploring.csv', index=False)
    return 'explore data is done'


