#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/30 10:00
# @Author  : Jun
# @File    : exec_progress.py

import numpy as np
import pandas as pd
from py_code.woe import *
import os
import re
from py_code.binning import verify_woe
from py_code.model import *
from py_code.func import *
from py_code.drawwoe import *


def data_split(df_,tag,test_size):
    from sklearn import model_selection
    df_y = df_[tag]
    df_x = df_.drop(tag, axis=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        df_x, df_y, test_size=test_size, random_state=128)
    df_train = pd.concat([x_train, y_train], axis=1)
    df_map = pd.concat([x_test, y_test], axis=1)
    return df_train,df_map


def assign_num_woe(x, dics, var_name):
    dic_w = dics[var_name]
    try:
        bin_null = [(bin_g, str(value)) for bin_g, value in dic_w.items() if ('-999' in bin_g) or ('-999.0' in bin_g)]
        bin_group_tmp, y_tmp = bin_null[0]
    except:
        bin_group_tmp = '(-999.0,-999.0)'
        y_tmp = str(0)
    for i in dic_w.keys():
        if str(x) in i and str(x) in ['-999', '-999.0']:
            y = dic_w[i]
            bin_group = i
        else:
            bin_num_1 = float(i.split(',')[0][1:])
            bin_num_1_f = i.split(',')[0][0]
            bin_num_2_f = i.split(',')[1][-1]
            try:
                bin_num_2 = float(i.split(',')[1][:-1])
            except ValueError:
                bin_num_2 = '-'
            if bin_num_1_f == '[' and bin_num_2_f == ']' and float(bin_num_1) <= float(x) <= float(bin_num_2):
                y = str(dic_w[i])
                bin_group = i
            elif bin_num_1_f == '(' and bin_num_2_f == ']' and float(bin_num_1) < float(x) <= float(bin_num_2):
                y = str(dic_w[i])
                bin_group = i
            elif bin_num_1_f == '(' and bin_num_2_f == ')' and bin_num_2 == '-' and float(bin_num_1) < float(x):
                y = str(dic_w[i])
                bin_group = i
    try:
        return y, bin_group
    except:
        y = y_tmp
        bin_group = bin_group_tmp
        return y, bin_group


def assign_char_woe(x,dics, var_name):
    dic_w = dics[var_name]
    x = str(x)
    try:
        bin_null = [(bin_g, str(value)) for (bin_g, value) in dic_w.items() if ('_null_' in bin_g)]
        bin_group_tmp, y_tmp = bin_null[0]
    except:
        bin_group_tmp = '(_null_,_null_)'
        y_tmp = str(0)
    seq_list = []
    word_list = []
    keys_list = list(dic_w.keys())
    for i, j in enumerate(keys_list):
        try:
            word_s = re.search('(\w.*\w)', j)
            word_sg = word_s.group(1)
            string_split = word_sg.split(',')
        except:
            string_split = j[1:-1]
        string_split_list = list(set(string_split))
        seq_list.extend([i]*len(string_split_list))
        word_list.extend(string_split_list)
    if x in word_list:
        x_loc = seq_list[word_list.index(x)]
        bin_group = keys_list[x_loc]
        y = str(dic_w[bin_group])
    else:
        y = y_tmp
        bin_group = bin_group_tmp
    return y, bin_group


def apply_woe(data, dic_woe, dic_type, save_path, key, target, cols=None):
    data = data.copy()
    if cols:
        vars_list = cols
    else:
        vars_list = data.columns.tolist()
    vars_list_ = vars_list[:]
    # vars_list_ = list(set(vars_list).union(set([key, target])))
    data_df = data.ix[:, [key, target]]
    for col in vars_list_:
        # try:
        if col in dic_woe.keys():
            if dic_type[col] == 'NUMBER':
                data[col].fillna(-999, inplace=True)
                data_df[col + '_woe'] = data[col].map(lambda x: assign_num_woe(x, dic_woe, col)[0])
                data_df[col + '_bin'] = data[col].map(lambda x: assign_num_woe(x, dic_woe, col)[1])
                print('WOE IS FINISHED', col)
            if dic_type[col] == 'STRING':
                data[col].fillna('_null_', inplace=True)
                data_df[col + '_woe'] = data[col].map(lambda x: assign_char_woe(x, dic_woe, col)[0])
                data_df[col + '_bin'] = data[col].map(lambda x: assign_char_woe(x, dic_woe, col)[1])
                print('WOE IS FINISHED', col)
            data_df[col] = data.ix[:, col]
        # except Exception as e:
        #     print(col, 'error mapped')
    return data_df


def func_good(x):
    if x==1:
        y=0
    else:
        y=1
    return y


def func_bad(x):
    if x==0:
        y=0
    else:
        y=1
    return y


def cal_ks_tt(df_rs, bin, tag):
    df_rs.ix[:,'good']=df_rs[tag].map(lambda x:func_good(x))
    df_rs.ix[:,'bad']=df_rs[tag].map(lambda x:func_bad(x))
    df_gp=df_rs.groupby([bin])['good','bad'].sum()
    try:
        df_gp.reset_index(bin, inplace=True)
        df_gp['test'] = df_gp[bin].apply(lambda x: float(x.split(',')[0][1:]))
        df_gp.sort_values(by='test', ascending=True, inplace=True)
        df_gp.drop('test', axis=1, inplace=True)
        df_gp.set_index(bin, inplace=True)
    except:
        df_gp.set_index(bin, inplace=True)
        df_gp['bad_rate'] = df_gp['bad']/(df_gp['bad'] + df_gp['good'])
        df_gp.sort_values(by='bad_rate',ascending=True,inplace=True)
        df_gp.drop('bad_rate', axis=1, inplace=True)
    bad_sum=df_gp['bad'].sum()
    good_sum=df_gp['good'].sum()
    bad_cnt=df_gp['bad'].cumsum()
    good_cnt=df_gp['good'].cumsum()
    df_gp.ix[:,'bad_cnt']=bad_cnt
    df_gp.ix[:,'good_cnt']=good_cnt
    df_gp['pct_bin']=(df_gp['good']+df_gp['bad'])/(bad_sum+good_sum)
    df_gp['pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp['bad_pct']=df_gp['bad']/np.array([bad_sum for _ in range(len(df_gp))])
    df_gp['good_pct']=df_gp['good']/np.array([good_sum for _ in range(len(df_gp))])
    df_gp['Woe']=np.log(df_gp['good_pct']/df_gp['bad_pct'])
    if 'inf' in str(df_gp['Woe'].tolist()):
        df_gp['Woe'] = df_gp['Woe'].map(lambda x: verify_woe(x))
    df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    df_gp.ix[:,'b_c_p']=df_gp['bad_cnt']/np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'g_c_p']=df_gp['good_cnt']/np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:,'ks']=df_gp['g_c_p']-df_gp['b_c_p']
    ks_max=max(df_gp['ks'].map(lambda x :abs(x)))
    iv=sum(df_gp['IV'].tolist())
    return ks_max,iv,df_gp[['good','bad','pct_bin','pct_default','Woe','IV','ks']]


def var_woe_estimate(train,test,train_cols,save_path,target):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writer = ExcelWriter(save_path + '/dev_oot_vars_desc.xlsx')
    train_i=train.copy()
    test_i=test.copy()
    varname_list=[]
    varks_list=[]
    ks_j_=[]
    ks_i_=[]
    iv_i_=[]
    iv_j_=[]
    check=[]
    group=[]
    num=0
    # train_cols = train_cols[:1]
    for i in train_cols:
        print('the var to estimate: ', i)
        if i in train.columns:
            ks_i, iv_i, df_gp1 = cal_ks_tt(train_i, i, target)
            ks_j, iv_j, df_gp2 = cal_ks_tt(test_i, i, target)
            varname_list.append(i)
            varks_list.append(abs(ks_i - ks_j))
            ks_j_.append(ks_j)
            ks_i_.append(ks_i)
            iv_i_.append(iv_i)
            iv_j_.append(iv_j)
            group.append(df_gp1.shape[0])
            df_describle = pd.concat([df_gp1, df_gp2], axis=1, join_axes=[df_gp1.index], keys=['TRAIN', 'TEST'])
            tt_idx = [idx for idx in df_describle.index if (idx.find('-999') < 0 and idx.find('_null_') < 0)]
            df_describle_tmp = df_describle[df_describle.index.isin(tt_idx)]
            df_describle_tmp = df_describle_tmp.sort_values(('TRAIN', 'Woe'))
            test_woe = df_describle_tmp['TEST']['Woe'].tolist()
            if pd.Series(test_woe).is_monotonic_decreasing or pd.Series(test_woe).is_monotonic_increasing:
                check.append(0)
            else:
                check.append(1)
            df_describle.to_excel(writer, startrow=num)
            num += len(df_describle) + 4
    test_ks = pd.DataFrame({'var': varname_list,
                            'ks_train': ks_i_,
                            'ks_test': ks_j_,
                            'ks_dif': varks_list,
                            'iv_train': iv_i_,
                            'iv_test': iv_j_,
                            'check': check,
                            'group': group})
    ks_sort = test_ks.sort_values('ks_test', ascending=False)[[
        'var', 'iv_train', 'iv_test', 'ks_train', 'ks_test', 'ks_dif', 'group', 'check']]
    ks_sort.to_excel(writer, 'summary', startrow=0)
    writer.save()
    return ks_sort


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


def exec_progress(data_pt, key, target, save_path, exclude_var_list=[]):
    if isinstance(data_pt, str):
        data = read_path(data_pt, ',')
    else:
        data = data_pt.copy()
    exclude_var_list = convert_lower(exclude_var_list)
    target = convert_lower(target)
    key = convert_lower(key)
    dev, oot = data_split(data, target, 0.3)
    # dic_woe, dic_df, dic_type = woe(dev, target, save_path, exclude_var_list)
    import pickle
    pik_file = open('/Users/jun/home/data/logistical_regression/output/VARIABLE_WOE.pkl', 'rb')
    dic_woe = pickle.load(pik_file)
    pik_file.close()
    pik_file = open('/Users/jun/home/data/logistical_regression/output/VARIABLE_TYPE.pkl', 'rb')
    dic_type = pickle.load(pik_file)
    pik_file.close()

    info = pd.read_excel(save_path + '/iv_sheet.xlsx', sheet_name='summary')
    keep_vars_list = info.sort_values('iv', ascending=False)[:3000]['variable'].tolist()
    keep_vars_list.extend([key, target])
    # print('train data starts to apply woe  ...')
    # m_data_train = apply_woe(dev[keep_vars_list], dic_woe, dic_type, save_path, key, target)
    # print('dev data starts to apply woe  ...')
    # m_data_test = apply_woe(oot[keep_vars_list], dic_woe, dic_type, save_path, key, target)

    print('start to save data ...')
    # m_data_train.to_csv(save_path + '/train_data.csv', index=False)
    # m_data_test.to_csv(save_path + '/test_data.csv', index=False)

    print('start ttdescribe ...')
    m_data_train = pd.read_csv(save_path + '/train_data.csv')
    m_data_test = pd.read_csv(save_path + '/test_data.csv')
    cols_bin = [i for i in m_data_train.columns if '_bin' in i]
    m_data_train = m_data_train[m_data_train[target].isin([0, 1])]
    m_data_test = m_data_test[m_data_test[target].isin([0, 1])]
    ks_sort = var_woe_estimate(m_data_train, m_data_test, cols_bin, save_path, target)

    ks_sort = pd.read_excel(save_path + '/dev_oot_vars_desc.xlsx', sheet_name='summary')
    nd_var = [i for i in info[info['iv'] > 0.01]['variable'].tolist()]
    ks_sort_k = [i[:i.find('_bin')] for i in ks_sort[(ks_sort['check'] == 0) & (ks_sort['ks_dif'] < 0.02)]['var'].tolist()]
    nd_var = list(set(nd_var).intersection(set(ks_sort_k)))
    vars_cols = filter_by_corr(m_data_train[[i + '_woe' for i in nd_var] + [key, target]], nd_var, corr_limit=0.75)
    vars_cols_df = pd.DataFrame(vars_cols, columns=['variable'])
    vars_cols_df.to_csv(save_path + '/variable_filter.csv', index=False)
    return m_data_train, m_data_test, vars_cols


def func_report(data,result,tag,B,A,save_path,test,info=None):
    '''
    :param data: Train
    :param result: 模型训练结果
    :param tag:
    :param base_score: 550
    :param double_score: 40
    :param save_path: 存储路径，默认当前目录下，不带.xlsx
    :param test: Test
    :param info: info表的detail页签
    :return:
    '''
    from pandas import ExcelWriter
    writer=ExcelWriter(save_path+'.xlsx')
    data['intercept'] = 1
    rs=data.copy()
    Y=data[tag]
    print('stage1 : LOGISTIC RESULT:')
    # print(result.summary())
    train_col = result.params.index.tolist()
    p = result.predict(data[train_col]).tolist()
    auc = roc_auc_score(Y, p)
    rs['p']=p
    # odds=float(len(Y[Y==0]))/float(len(Y[Y==1]))
    odds=30
    print('stage1.5:PREPARE VAR RESULT：')
    num=0
    tt = var_woe_estimate(data, test, [i[:-4]+'_bin' for i in train_col if '_woe' in i], save_path[:-5], tag)
    print('stage3: BEGIN TO CAL SCORE_I:()')
    # df_rs_score1 = cal_score(rs, base_score, double_score, odds)
    df_rs_score1,B,A = cal_score(rs, B=B, A=A)
    # df_rs_score1.to_excel(writer,sheetname='score_detail')
    print('stage4: BEGIN TO CAL KS ：')
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score1,tag,'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score1,tag,'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score1,tag,'qcut')
    ks=[auc,ks_max_uncut,ks_max_cut,ks_max_qcut]
    df_gp_cut.to_excel(writer,'train_gp_score',startrow=0)
    df_gp_qcut.to_excel(writer,'train_gp_score',startrow=df_gp_cut.shape[0]+3)
    ks_df=pd.DataFrame(ks,columns=['describe'],index=['auc','ks_max_uncut','ks_max_cut','ks_max_qcut'])
    ks_df.to_excel(writer,'result_summary',startrow=0)
    df_gp_uncut2, df_gp_cut2,df_gp_qcut2, ks_df2,p2,df_rs_score2=cal_ks_test(result,test,tag,B,A)
    df_gp_cut2.to_excel(writer,'test_gp_score',startrow=0)
    df_gp_qcut2.to_excel(writer,'test_gp_score',startrow=df_gp_cut.shape[0]+3)
    ks_df2.to_excel(writer, 'result_summary', startrow=ks_df.shape[0]+3)

    df_score=df_rs_score1.append(df_rs_score2)
    ks, gp = gb_add_woe(df_score, tag, '10score_cut')
    gp.to_excel(writer,'all_data_score_10cut',startrow=0)
    result_df=pd.DataFrame({'params':list(result.params.index),
                            'coef':list(result.params),
                            'p_value':list(result.pvalues)})
    result_df.to_excel(writer,'logistic_function')
    writer.save()
    if info:
        var_list=[i[:-4] for i in result.params.index if '_woe' in i]
        draw_woe(info, var_list, save_path)
    print('--------AUC: '+str(auc)+'---------')
    print('---------------KS: -----------------')
    print(ks_df)
    print('----------------------------------')
    print(df_gp_cut)
    # draw_roc([[Y, p], [test[tag], p2]],['train','test'])
    return df_score


