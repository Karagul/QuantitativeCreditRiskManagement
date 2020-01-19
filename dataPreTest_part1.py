# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:48:25 2020

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


path = 'gt_'
raw_data_file_name = 'modify_data.csv'
basic_check = True
smy_creation = True
#检验文件基本属性

if basic_check:
    badsmp = FeatureStatTools.smp_valid_check(path+'/'+raw_data_file_name, path+'/stat')
    if len(badsmp['line'])>0:
        FeatureStatTools.badSmpRm(path+'/'+raw_data_file_name, path+'/raw_features.csv', list(badsmp['line']))
#检查特征类型
    type_check = FeatureStatTools.ft_type_check(path+'/'+raw_data_file_name, path, header = True, size_c = 1000)

if smy_creation:
    raw_data = pd.read_csv(path+'/'+raw_data_file_name, header = 0)
    to_drop_list = ['num']
    smy={'label':'label','dayno':'back_time','int_col':[],'float_col':[],'str_col':[],'toDrop':[]}
    total_data = len(raw_data)
    try:
        js=tools.getJson(path+'/'+'type_info.json')
    except:
        js=tools.getJson(path+'/'+'type_info_sample.json')
    print('generating summary')
    for k,v in tqdm(js.items()):
        if k==smy['label'] or k == smy['dayno']:
            continue
        elif k in to_drop_list:
            smy['toDrop'].append({k:'no feature'})
        elif js[k]['dist']<=1:
            smy['toDrop'].append({k:'unique_value'})
        elif js[k]['type']=='str' and js[k]['dist']>30:
            smy['toDrop'].append({k:'too much chars'})
        elif raw_data[k].isnull().sum()/total_data>0.98:
            smy['toDrop'].append({k:'too much missing'})
        elif js[k]['type']=='str':
            smy['str_col'].append(k)
        elif js[k]['type']=='int':
            smy['int_col'].append(k)
        elif js[k]['type']=='float':
            smy['float_col'].append(k)
    
    tools.putFile(path,'summary.json',smy)