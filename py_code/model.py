#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 16:08
# @Author  : Jun
# @File    : model.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import math
from pandas import ExcelWriter


def filter_by_corr(data_df, series_, corr_limit=0.75):
    both_var = [i + '_woe' for i in series_ if i + '_woe' in data_df.columns]
    drop_set, both_var_set = set(), set(both_var)
    for var in both_var:
        if var in both_var_set:
            both_var_set.remove(var)
        if var not in drop_set:
            drop_set |= {v for v in both_var_set if abs(np.corrcoef(
                data_df[var].values, data_df[v].values)[0][1]) > corr_limit}
            both_var_set -= drop_set
    return [i for i in both_var if i not in drop_set]


def func_sort_col(col, data_df, target):
    cols = col[:]
    data_df = data_df.copy()
    if 'intercept' not in cols:
        cols.append('intercept')
        data_df['intercept'] = 1
    cols = list(set(cols).intersection(set(data_df.columns)))
    logit_1 = sm.Logit(data_df[target], data_df[cols])
    result_1 = logit_1.fit()
    wald_chi2 = np.square(result_1.params / np.square(result_1.bse))
    a = pd.DataFrame(wald_chi2, columns=['value'])
    b = a.sort_values('value', ascending=False)
    sorted_cols = b.index.tolist()
    sorted_cols.remove('intercept')
    a = a[a.index.isin(sorted_cols)]
    return a


def func_good(x):
    if x == 1:
        y = 0
    else:
        y = 1
    return y


def func_bad(x):
    if x == 0:
        y = 0
    else:
        y = 1
    return y


def catch_score(x, group):
    for i in group:
        if float(eval(i)[0]) < x <= float(eval(i)[1]):
            bin = i
    try:
        return bin
    except:
        print(x, eval(i)[0])
        bin = 0
        return bin


def group_score(data,code,cut,score_name=None):
    '''
    :param data:
    :param code:
    :param cut: qcut/cut/10score_cut/uncut
    :param score_name: 可选填，当score的名字自定义非'score'时
    :return:
    '''
    df_rs = data.copy()
    list = df_rs.index.tolist()
    df_rs.ix[:, 'no'] = list
    if score_name:
        df_rs['score']=df_rs[score_name]
    try:
        if cut == 'qcut':
            a = pd.qcut(df_rs['score'], 10)
            df_rs['score_group']=a
        elif cut == 'cut':
            a = pd.cut(df_rs['score'], 10)
            df_rs['score_group'] = a
        elif cut == 'uncut':
            df_rs['score_group'] = df_rs['score'][:]
        elif cut == '10score_cut':
            group = ['['+str(i)+','+str(i+10)+']' for i in range(100,1300,10)]
            df_rs['score_group']=df_rs['score'].map(lambda x: catch_score(x,group))
        elif cut == '2score_cut':
            group = ['['+str(i)+','+str(i+10)+']' for i in range(100,1300,2)]
            df_rs['score_group']=df_rs['score'].map(lambda x: catch_score(x,group))
        else:
            print('error! check cut_value，cut_method changed to uncut (⊙v⊙) ')
            df_rs['score_group'] = df_rs['score'][:]
    except:
        print('error! check cut_value，cut_method changed to uncut (⊙v⊙) ')
        df_rs['score_group'] = df_rs['score'][:]
    df_rs['good'] = df_rs[code].map(lambda x: func_good(x))
    df_rs['bad'] = df_rs[code].map(lambda x: func_bad(x))
    b = df_rs.groupby(['score_group', ])['no', 'good', 'bad'].sum()
    df_gp = pd.DataFrame(b)
    # print df_gp
    return df_gp


def gb_add_woe(data,code,cut,score_name=None):
    if score_name:
        df_gp = group_score(data, code, cut,score_name)
    else:
        df_gp=group_score(data,code,cut)
    df_gp.ix[:,'pct_default']=df_gp['bad']/(df_gp['bad']+df_gp['good'])
    bad_sum=df_gp['bad'].sum()
    good_sum=df_gp['good'].sum()
    df_gp.ix[:,'bad_pct']=df_gp['bad']/np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'good_pct']=df_gp['good']/np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'odds']=df_gp['good_pct']-df_gp['bad_pct']
    df_gp.ix[:,'Woe']=np.log(df_gp['good_pct']/df_gp['bad_pct'])
    df_gp.ix[:,'IV']=(df_gp['good_pct']-df_gp['bad_pct'])*df_gp['Woe']
    bad_cnt=df_gp['bad'].cumsum()
    good_cnt=df_gp['good'].cumsum()
    df_gp.ix[:,'bad_cnt']=bad_cnt
    df_gp.ix[:,'good_cnt']=good_cnt
    df_gp.ix[:,'b_c_p']=df_gp['bad_cnt']/np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'g_c_p']=df_gp['good_cnt']/np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'KS']=df_gp['g_c_p']-df_gp['b_c_p']
    df_gp['bin']=list(df_gp.index.map(lambda x :str(x)+')'))[:]
    df_gp['bin_pct']=(df_gp['good']+df_gp['bad'])/(bad_sum+good_sum)
    df_gp=df_gp.reset_index(drop=True)
    ks_max=df_gp['KS'].max()
    # print(df_gp)
    return ks_max,df_gp


def cal_score(data,base_score=None,double_score=None,odds=None,B=None,A=None):
    '''
    :param data: df_['id','tag'] 为了方便要统一一下字段名字_(:зゝ∠)_
    :param odds:  全局bad/good
    :param result:模型训练结果，以result格式
    :return:每个客户的总分
    '''
    if B is None:
        B = double_score / math.log(2)
        A = base_score + B * math.log(odds)
    df_rs=data.copy()
    df_rs.ix[:, 'p_'] = df_rs['p'].map(lambda x:1-float(x))
    df_rs.ix[:, 'score'] = np.array([A for _ in range(len(df_rs))]) + \
                           np.array([B for _ in range(len(df_rs))]) * np.log(df_rs.ix[:, 'p'] / df_rs.ix[:, 'p_'])
    df_rs['score'].astype(int)
    return df_rs, B, A


def cal_ks(result1, data, tag, base_score, double_score):
    '''
    :param data: 客户数据['id','tag']
    :param X: data[train_cols]
    :param Y: data[tag]
    :param base_score:
    :param woe_dic:m_data所返回的变量woe字典
    :param v: odds
    :return: 根据已选出来的变量，生成模型结果报告
    '''
    rs = data.copy()
    Y = data[tag]
    # print(result1.summary())
    p = result1.predict().tolist()
    auc = roc_auc_score(Y, p)
    rs['p'] = p
    odds = float(len(Y[Y == 0]))/float(len(Y[Y == 1]))
    # print 'stage3: 开始计算单个客户得分：'
    df_rs_score, B, A = cal_score(rs, base_score=base_score, double_score=double_score, odds=odds)
    # print 'stage4: 开始计算KS：'
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score, tag, 'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score, tag, 'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score, tag, 'qcut')
    ks=[auc,ks_max_uncut,ks_max_cut,ks_max_qcut]
    ks_df=pd.DataFrame(ks,columns=['describe'],index=['auc','ks_max_uncut','ks_max_cut','ks_max_qcut'])
    print('--------AUC: '+str(auc)+'---------')
    print('---------------KS: -----------------')
    print(ks_df)
    return ks_df, B, A


def func_stepwise_1(cols, data_df, target, report=None):
    vars_dic = {}
    train_cols = ['intercept']
    data_df['intercept'] = 1
    error_var = []
    sorted_cols = cols[:]
    num = 0
    if report:
        logit = sm.Logit(data_df[target], data_df[cols])
        result = logit.fit()
    else:
        for i in sorted_cols:
            print('has looped to ' + str(i))
            train_cols = list(set(train_cols) | set(['intercept', i]))
            try:
                ori = train_cols[:]
                logit = sm.Logit(data_df[target], data_df[ori])
                result = logit.fit()
                train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
                train_params = result.params[result.params > 0].index.tolist()
                train_cols = list(set(train_p) - set(train_params))
            except Exception as e:
                print(e)
                train_cols.remove(i)
                error_var.append(i)
    train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
    train_params = result.params[result.params > 0].index.tolist()
    train_cols = list(set(train_p) - set(train_params))
    logit = sm.Logit(data_df[target], data_df[train_cols])
    result = logit.fit()
    # print(result.summary())
    print('aic： ' + str(result.aic))
    # p = result.predict().tolist()
    # Y = data_df[target]
    # auc = roc_auc_score(Y, p)
    ks_df, B, A = cal_ks(result, data_df, target, 550, 40)
    return train_cols, result, vars_dic, ks_df.ix[1, 'describe'], error_var,B , A


# def cal_ks_test(result, test_data, tag, B, A):
#     '''
#     :param data: 客户数据['id','tag']
#     :param X: data[train_cols]
#     :param Y: data[tag]
#     :param base_score:
#     :param woe_dic:m_data所返回的变量woe字典
#     :param v: odds
#     :return: 根据已选出来的变量，生成模型结果报告
#     '''
#
#     rs = test_data.copy()
#     predict_cols = result.params.index.tolist()
#     rs['intercept'] = 1.0
#     p = result.predict(rs[predict_cols])
#     rs['p'] = p
#     Y = rs[tag]
#     auc = roc_auc_score(Y, p)
#     df_rs_score,B,A = cal_score(rs, B=B, A=A)
#     ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score, tag, 'uncut')
#     ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score, tag, 'cut')
#     ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score, tag, 'qcut')
#     ks = [auc, ks_max_uncut, ks_max_cut, ks_max_qcut]
#     ks_df = pd.DataFrame(ks, columns=['describe'],index=['auc', 'ks_max_uncut', 'ks_max_cut', 'ks_max_qcut'])
#     print('--------AUC: '+str(auc)+'---------')
#     print('---------------KS: -----------------')
#     print(ks_df)
#     print('---------------cut:-----------------')
#     # print (df_gp_cut)
#     # return df_rs_score
#     return df_gp_uncut, df_gp_cut,df_gp_qcut, ks_df, p, df_rs_score


def cal_ks_test(result,test_data,tag,base_score=550,double_score=40,odds=30):
    '''
    :param data: 客户数据['id','tag']
    :param X: data[train_cols]
    :param Y: data[tag]
    :param base_score:
    :param woe_dic:m_data所返回的变量woe字典
    :param v: odds
    :return: 根据已选出来的变量，生成模型结果报告
    '''
    rs = test_data.copy()
    predict_cols = result.params.index.tolist()
    rs['intercept'] = 1.0
    p = result.predict(rs[predict_cols])
    rs['p'] = p
    Y=rs[tag]
    auc = roc_auc_score(Y, p)
    df_rs_score,B,A = cal_score(rs, base_score=base_score, double_score=double_score, odds=odds)
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score,tag,'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score,tag,'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score,tag,'qcut')
    ks=[auc,ks_max_uncut,ks_max_cut,ks_max_qcut]
    ks_df=pd.DataFrame(ks,columns=['describe'],index=['auc','ks_max_uncut','ks_max_cut','ks_max_qcut'])
    print('--------AUC: '+str(auc)+'---------')
    print('---------------KS: -----------------')
    print(ks_df)
    print('---------------cut:-----------------')
    # print (df_gp_cut)
    # return df_rs_score
    return df_gp_uncut, df_gp_cut,df_gp_qcut, ks_df,p,df_rs_score