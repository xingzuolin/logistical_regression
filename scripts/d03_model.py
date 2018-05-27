#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 15:35
# @Author  : Jun
# @File    : d02_cal_woe.py


from py_code.exec_progress import *
import numpy as np
import pandas as pd
import argparse


def par():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt_pth', help="the base data")
    parser.add_argument('--dt_lbl', help="the base label")
    parser.add_argument('--dt_opt', help="the output")
    args=parser.parse_args()
    return args


exclude_var_list = ['apply_no','DOLL_WEIGHT','SEG','apply_amt', 'loan_terms_days']
target = 'acquisition_is_bad'
if __name__ == "__main__":
    # args = par()
    # dt_output = args.dt_opt
    dt_output = '/Users/jun/home/data/logistical_regression/output'
    vars_cols_df = pd.read_csv(dt_output + '/variable_filter.csv')
    vars_cols = np.array(vars_cols_df).transpose().tolist()[0]
    m_data_train = pd.read_csv(dt_output + '/train_data.csv')
    m_data_test = pd.read_csv(dt_output + '/test_data.csv')
    m_data_train = m_data_train[m_data_train[target].isin([0, 1])]
    m_data_test = m_data_test[m_data_test[target].isin([0, 1])]

    # vars_cols = vars_cols[:5]

    sorted_cols = func_sort_col([i for i in vars_cols], m_data_train[[i for i in vars_cols]+['apply_no', target]], target)
    srt_cols = sorted_cols.sort_values('value', ascending=False).index.tolist()
    srt_cols_tmp = [i[:-4]+'_bin' for i in srt_cols]
    srt_cols.extend(srt_cols_tmp)
    print(srt_cols)
    train_cols, result, vars_dic, ks_df1, error_var,B,A = func_stepwise_1(srt_cols, m_data_train, target)
    print('===============================================')
    df_gp_uncut, df_gp_cut, df_gp_qcut, ks_df, p, df_rs_score = cal_ks_test(result, m_data_test, target,550,40,30)
    print(m_data_train[target].value_counts(normalize=True))
    print(m_data_test[target].value_counts(normalize=True))
    # df_gp_uncut, df_gp_cut, df_gp_qcut, ks_df, p, df_rs_score = cal_ks_test(result, m_data_test, target, B=B, A=A)
    # df_score2 = func_report(m_data_train[srt_cols+['apply_no','acquisition_is_bad']], result, 'acquisition_is_bad', B,A, dt_output+'/sj_uscore_model', m_data_test[srt_cols+['apply_no','acquisition_is_bad']])