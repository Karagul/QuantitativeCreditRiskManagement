# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:24:51 2020

@author: zhuchang
"""
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

import tools
from WoeMethods import AllWoeFuncs, WoeFuncs
import FeatureStatTools 

path = 'lizi_data'
raw_data_file_name = 'raw_data.csv'

#读取原始数据
try:
    type_check = tools.getJson(path+'/'+'type_info.json')
except:
    type_check = tools.getJson(path+'/'+'type_info_sample.json')
raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = ',', header = 0, dtype = {i:type_check[i]['type'] for i in type_check.keys() if type_check[i]['type'] == 'str'})
raw_data = raw_data[raw_data['ft_dev_phone_brand'] != 'Apple']
#summary.json is required, indicating modelers primary knowledge abouut features
print("Reading related information...")
smy = tools.getJson(path+'/summary.json')
toDrop = smy.get('toDrop')
toDropList = [list(a.keys())[0] for a in toDrop if list(a.values())[0] != 'no feature']
#ids = smy.get('ids')
str_col = smy.get('str_col')
int_col = smy.get('int_col')
float_col = smy.get('float_col')
#toOneHot = smy.get('toOneHot')
dayno = smy.get('dayno')
label = smy.get('label')
"""取日期中点为单因子统计做准备"""
#dayno_mid = raw_data[dayno].median()
dayno_mid = 20190907

ivs = {i:{} for i in str_col+float_col+int_col}; kses = {i:{} for i in str_col+float_col+int_col}
tvs = {i:{} for i in str_col+float_col+int_col}; psis = {i:{} for i in str_col+float_col+int_col}
mis = {i:{} for i in str_col+float_col+int_col+toDropList}

#单因子检验的时候均不考虑缺失值，除非是缺失值的统计
print("--------------缺失及分布情况简易统计----------------------")
try:
    with tqdm(int_col+float_col+str_col+toDropList) as t:
        for i in t:
            #缺失情况统计
            mis[i] = FeatureStatTools.ft_mis_check2(raw_data[[i, label]], type_check[i]['type'])[i]
except KeyboardInterrupt:
    t.close()
    raise
t.close()

print("--------------单因子逻辑回归----------------------")
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            df_bf, df_af = raw_data[raw_data[dayno]<=dayno_mid][[i, 'label']], raw_data[raw_data[dayno]>dayno_mid][[i, 'label']]
            if type_check[i]['type'] != 'str':
                #t值计算
                try:
                    tvs[i] = {'bf':FeatureStatTools.tvalue_cal_func(df_bf[:], ifconst = True),\
                              'af':FeatureStatTools.tvalue_cal_func(df_af[:], ifconst = True)}
                except:
                    tvs[i] = {'bf':None, 'af':None}
                    
            else:
                tvs[i] = {'bf':None, 'af':None}
except KeyboardInterrupt:
    t.close()
    raise
t.close()

print("--------------KS计算----------------------")
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            df_bf, df_af = raw_data[raw_data[dayno]<=dayno_mid][[i, 'label']], raw_data[raw_data[dayno]>dayno_mid][[i, 'label']]
            if type_check[i]['type'] != 'str':
                #KS计算
                try:
                    kses[i] = {'bf':FeatureStatTools.ks_cal_func(df_bf[:], grps=10, ascd = False, duplicates = 'drop')['ks'].apply(abs).max(),\
                               'af':FeatureStatTools.ks_cal_func(df_af[:], grps=10, ascd = False, duplicates = 'drop')['ks'].apply(abs).max()}
                except:
                    kses[i] = {'bf':None, 'af':None}
            else:
                kses[i] = {'bf':None, 'af':None}
except KeyboardInterrupt:
    t.close()
    raise
t.close()

print("--------------PSI计算----------------------")
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            df_bf, df_af = raw_data[raw_data[dayno]<=dayno_mid][[i, 'label']], raw_data[raw_data[dayno]>dayno_mid][[i, 'label']]
            if type_check[i]['type'] != 'str':
                #psi计算
                try:
                    psis[i] = FeatureStatTools.psi_cal_func(df_bf[[i]], df_af[[i]], grps = 10)
                except:
                    psis[i] = None
            else:
                psis[i] = None
                
except KeyboardInterrupt:
    t.close()
    raise
t.close()

print("--------------IV计算----------------------")
spurs = WoeFuncs(pct_size = 0.02, max_grps = 5, chiq_pv = 0.05, ifmono = False, keepnan = True, methods = 'tree')
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            df_bf, df_af = raw_data[raw_data[dayno]<=dayno_mid][[i, 'label']], raw_data[raw_data[dayno]>dayno_mid][[i, 'label']]
            #IV计算
            #if i in str_col:
            if type_check[i]['type'] == 'str':
                try:
                    tmp = {}
                    spurs.setTgt(df_bf)
                    spurs.strWoe_cal()
                    tmp['bf'] = spurs.getIVinfo()
                    spurs.setTgt(df_af)
                    spurs.strWoe_cal()
                    tmp['af'] = spurs.getIVinfo()
                    ivs[i] = tmp
                except:
                    ivs[i] = {'bf':None, 'af':None}
            elif type_check[i]['type'] in ['int', 'float']:
                try:
                    tmp = {}
                    spurs.setTgt(df_bf)
                    spurs.woe_cal()
                    tmp['bf'] = spurs.getIVinfo()
                    spurs.setTgt(df_af)
                    spurs.woe_cal()
                    tmp['af'] = spurs.getIVinfo()
                    ivs[i] = tmp
                    
                except:
                    ivs[i] = {'bf':None, 'af':None}

except KeyboardInterrupt:
    t.close()
    raise
t.close()
#输出不适合计算WOE的特征
#print(spurs.allInvalid)
tools.putFile(path+'/feature_stat', 'invalidIV_mdf.json', spurs.getInvalid())
tools.putFile(path+'/feature_stat','misStat_mdf.json', mis)
tools.putFile(path+'/feature_stat','ivStat_mdf.json', ivs)
tools.putFile(path+'/feature_stat','ksStat_mdf.json', kses)
tools.putFile(path+'/feature_stat','tvStat_mdf.json', tvs)
tools.putFile(path+'/feature_stat','psiStat_mdf.json', psis)

final = pd.DataFrame(mis).T
tmp = pd.DataFrame(ivs).T
tmp.columns = ['iv_'+str(a) for a in tmp.columns]
final = pd.merge(left = final, right = tmp, left_index = True, right_index = True, how = 'outer')
tmp = pd.DataFrame(kses).T
tmp.columns = ['ks_'+str(a) for a in tmp.columns]
final = pd.merge(left = final, right = tmp, left_index = True, right_index = True, how = 'outer')
tmp = pd.DataFrame(tvs).T
tmp.columns = ['tvalue_'+str(a) for a in tmp.columns]
final = pd.merge(left = final, right = tmp, left_index = True, right_index = True, how = 'outer')
tmp = pd.DataFrame(pd.Series(psis), columns = ['psi'])
final = pd.merge(left = final, right = tmp, left_index = True, right_index = True, how = 'outer')