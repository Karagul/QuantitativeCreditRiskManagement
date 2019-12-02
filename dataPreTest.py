import pandas as pd
import numpy as np
import os
import json

from tools import *
from WoeMethods import AllWoeFUncs, WoeFuncs
from FeatureProcessUnspv import FeatureProcessFuncs
from FeatureStatTools import *


#设置最基本的路径变量
path = './rawdata'
basic_check = True
raw_data_file_name = ''
#检验文件基本属性

if basic_check:
    badsmp = smp_valid_check(path+'/'+raw_data_file_name, path+'/basicStat_badsmp.csv', size_c = 1000)
    if len(badsmp)>0:
        badSmpRm(path+'/'+raw_data_file_name, path+'/raw_features', badsmp)
#检查特征类型
type_check = ft_type_check(path+'/'+raw_data_file_name, path+'/basicStat_typecheck', header = True, size_c = 1000)

#读取原始数据
raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = r'[_ \t,| ]', header = True, dtype = {i:type_check[i]['type'] for i in type_check.keys())

#summary.json is required, indicating modelers primary knowledge abouut features
print("Reading related information...")
smy = getJson(path+'/summary.json')
toDrop = smy.get('toDrop')
#ids = smy.get('ids')
str_col = smy.get('str_col')
int_col = smy.get('int_col')
float_col = smy.get('float_col')
#toOneHot = smy.get('toOneHot')
dayno = smy.get('dayno')
"""取日期中点为单因子统计做准备"""
dayno_mid = raw_data[dayno].median()

ivs = {i:{} for i in str_col+float_col+int_col}; kses = {i:{} for i in str_col+float_col+int_col}
tvs = {i:{} for i in str_col+float_col+int_col}; psis = {i:{} for i in str_col+float_col+int_col}
mis = {i:{} for i in str_col+float_col+int_col}

#单因子检验的时候均不考虑缺失值，除非是缺失值的统计
try:
    with tqdm(int_col+float_col+str_col) as t:
        for i in t:
            #缺失情况统计
            print("--------------缺失及分布情况简易统计----------------------"")
            mis[i] = ft_mis_check(raw_data[i], type_check[i]['type'])
            df_bf, df_af = raw_data[raw_data[dayno]<=dayno_mid][[i, 'label']], raw_data[raw_data[dayno]>dayno_mid][[i, 'label']]
            if type_check[i]['type'] != 'str':
                #t值计算
                try:
                    tvs[i] = {'bf':tvalue_cal_func(df_bf, ifconst = True), 'af':tvalue_cal_func(df_af, ifconst = True)}
                except:
                    tvs[i] = {'bf':None, 'af':None}

                #KS计算
                try:
                    kses[i] = {'bf':ks_cal_func(df_bf, grps=10, ascd = False)['ks'].max(), 'af':ks_cal_func(df_af, grps=10, ascd = False)['ks'].max()}
                except:
                    kses[i] = {'bf':None, 'af':None}

                #psi计算
                try:
                    psis[i] = psi_cal_func(df_bf, df_af, grps = 10)
                except:
                    psis[i] = None
            else:
                tvs[i] = {'bf':None, 'af':None}
                kses[i] = {'bf':None, 'af':None}
                psis[i] = None

            #IV计算
            spurs = WoeFuncs(pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
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
                    spurs.Woe_cal()
                    tmp['bf'] = spurs.getIVinfo()
                    spurs.setTgt(df_af)
                    spurs.Woe_cal()
                    tmp['af'] = spurs.getIVinfo()
                    ivs[i] = tmp
                except:
                    ivs[i] = {'bf':None, 'af':None}

except KeyboardInterrupt:
    t.close()
    raise
t.close()

putFile(path+'/feature_stat','misStat.json', mis)
putFile(path+'/feature_stat','ivStat.json', ivs)
putFile(path+'/feature_stat','ksStat.json', kses)
putFile(path+'/feature_stat','tvStat.json', tvs)
putFile(path+'/feature_stat','psiStat.json', psi)
