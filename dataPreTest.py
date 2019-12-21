import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

import tools
from WoeMethods import AllWoeFuncs, WoeFuncs
from FeatureProcess import FeatureProcessFuncs
import FeatureStatTools 


#设置最基本的路径变量
path = '../function_test/raw_data'
basic_check = False
raw_data_file_name = 'data.csv'
#检验文件基本属性

if basic_check:
    badsmp = FeatureStatTools.smp_valid_check(path+'/'+raw_data_file_name, path+'/stat')
    if len(badsmp['line'])>0:
        FeatureStatTools.badSmpRm(path+'/'+raw_data_file_name, path+'/raw_features.csv', list(badsmp['line']))
#检查特征类型
type_check = FeatureStatTools.ft_type_check(path+'/'+raw_data_file_name, path, header = True, size_c = 1000)

#读取原始数据
raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = r'[\t,|]', header = 0, dtype = {i:type_check[i]['type'] for i in type_check.keys() if type_check[i]['type'] == 'str'})

#summary.json is required, indicating modelers primary knowledge abouut features
print("Reading related information...")
smy = tools.getJson(path+'/summary.json')
toDrop = smy.get('toDrop')
#ids = smy.get('ids')
str_col = smy.get('str_col')
int_col = smy.get('int_col')
float_col = smy.get('float_col')
#toOneHot = smy.get('toOneHot')
dayno = smy.get('dayno')
label = smy.get('label')
"""取日期中点为单因子统计做准备"""
dayno_mid = raw_data[dayno].median()

ivs = {i:{} for i in str_col+float_col+int_col}; kses = {i:{} for i in str_col+float_col+int_col}
tvs = {i:{} for i in str_col+float_col+int_col}; psis = {i:{} for i in str_col+float_col+int_col}
mis = {i:{} for i in str_col+float_col+int_col}

#单因子检验的时候均不考虑缺失值，除非是缺失值的统计
print("--------------缺失及分布情况简易统计----------------------")
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            #缺失情况统计
            mis[i] = FeatureStatTools.ft_mis_check(raw_data[[i]], type_check[i]['type'])
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
                    tvs[i] = {'bf':FeatureStatTools.tvalue_cal_func(df_bf[:], ifconst = True), 'af':FeatureStatTools.tvalue_cal_func(df_af[:], ifconst = True)}
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
                    kses[i] = {'bf':FeatureStatTools.ks_cal_func(df_bf[:], grps=10, ascd = False)['ks'].max(), 'af':FeatureStatTools.ks_cal_func(df_af[:], grps=10, ascd = False)['ks'].max()}
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
spurs = WoeFuncs(pct_size = 0.02, max_grps = 5, chiq_pv = 0.05, ifmono = False, ifnan = True, methods = 'tree')
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
tools.putFile(path+'/feature_stat', 'invalidIV.json', spurs.getInvalid())
tools.putFile(path+'/feature_stat','misStat.json', mis)
tools.putFile(path+'/feature_stat','ivStat.json', ivs)
tools.putFile(path+'/feature_stat','ksStat.json', kses)
tools.putFile(path+'/feature_stat','tvStat.json', tvs)
tools.putFile(path+'/feature_stat','psiStat.json', psis)
