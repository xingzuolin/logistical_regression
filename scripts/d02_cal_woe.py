#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 15:35
# @Author  : Jun
# @File    : d02_cal_woe.py


from py_code.exec_progress import *
from py_code import func
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


exclude_var_list = ['apply_no','DOLL_WEIGHT','SEG','apply_amt', 'loan_terms_days', 'prob_scr_benchmark']
target = 'acquisition_is_bad'


if __name__ == "__main__":
    # args = par()
    # dt_output = args.dt_opt
    dt_output = '/Users/jun/home/data/logistical_regression/output'
    exclude_pth_num = dt_output + '/NumericVariablesExploring.csv'
    exclude_pth_chr = dt_output + '/ChracterVariablesExploring.csv'
    exclude_vars_p = func.vars_keep(exclude_pth_num, var_name='variable', method='miss_pct', proportion=0.85)
    exclude_var_list.extend(exclude_vars_p)
    exclude_vars_c = func.vars_keep(exclude_pth_chr, var_name='variable', method='miss_pct', proportion=0.85)
    exclude_var_list.extend(exclude_vars_c)
    m_data_train, m_data_test, vars_cols = exec_progress(data_pt=dt_output + '/dataset.csv',key='apply_no', target=target, save_path=dt_output, exclude_var_list=exclude_var_list)
    print(m_data_train.shape)
    print('--------------test-----------')
    print(m_data_test.shape)
    print('--------------the number of var---------------')
    print(len(vars_cols))

