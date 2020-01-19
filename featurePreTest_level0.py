# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:34:30 2020

@author: zhuchang
"""

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import re
import json
from tqdm import tqdm
from scipy import stats
import statsmodels.api as sm
import xgboost as xgb
import random
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold

import model_builder
import tools
from WoeMethods import bins_method_funcs, WoeFuncs
from FeatureProcess import AllFtrProcess
from FeatureSelection import ModelBasedMethods
import model_builder
import FeatureStatTools

#设置最基本的路径变量
if_gnrt_smy = True
ifselect = True
ifandriod = True
keep_emb = False
path = 'gt_big'
version = 'level0_gtMethods'
raw_data_file_name = 'raw_data.csv'
#此方法的调参细节
fittingpart = False
oottest = True
preSelect = True
#path = '../function_test/raw_data'
rnd_seed = 21

type_check = tools.getJson(path + '/type_info.json')
#raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = ',', header = 0, dtype = {i:type_check[i]['type'] for i in type_check.keys() if type_check[i]['type'] == 'str'})
raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = ',', header = 0)
"""
判断是否是andriod数据
"""
if ifandriod:
    raw_data = raw_data[raw_data['ft_dev_phone_brand']!='Apple']
    version = version+'_Andr'
else:
    version = version+'_Appl'
    
"""
读取特征相关的统计
"""
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

raw_data = raw_data.drop(toDropList, axis = 1)
    
"""
判断是否加入embedding特征
"""
if not keep_emb:
    embs = ['h'+str(a) for a in range(50)] + ['f'+str(a) for a in range(50)]
    raw_data = raw_data.drop(embs, axis = 1)
    str_col = list(set(str_col)-set(embs))
    int_col = list(set(int_col)-set(embs))
    float_col = list(set(float_col)-set(embs))
    version = version+'_nonEmb'
else:
    version = version+'_wthEmb'
    
if preSelect:
    print('------------------------------------特征覆盖情况预筛---------------------------------------')
    mis = {i:{} for i in str_col+float_col+int_col}
    fthr2Drop = []
    try:
        with tqdm(int_col+float_col+str_col) as t:
            for i in t:
                #缺失情况统计
                mis[i] = FeatureStatTools.ft_mis_check2(raw_data[[i, label]], type_check[i]['type'])[i]
                if mis[i]['type'] == 'int':
                    if mis[i]['cvr_rate'] < 0.05:
                        fthr2Drop += [i]
                    else:
                        if raw_data[i].value_counts().max()/raw_data[i].value_counts().sum() > 0.95:
                            fthr2Drop += [i]
                    
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    raw_data = raw_data.drop(fthr2Drop, axis = 1)
    str_col = list(set(str_col)-set(fthr2Drop))
    int_col = list(set(int_col)-set(fthr2Drop))
    float_col = list(set(float_col)-set(fthr2Drop))
    version = version + '_preSelect'
else:
    version = version + '_NonPreSelect'
    
if if_gnrt_smy:
    smy = {'undo':[], 'fill':{}, 'cap':{}, 'var2char':{}, 'onehot':{}, 'woeCal':{}}
    for i in int_col:
        smy['woeCal'][i] = {'type_info':'int'}
        
    for i in float_col:
        smy['woeCal'][i] = {'type_info':'float'}
        
    """
    对于字符串项特征的特殊处理
    """
    smy['woeCal']['ft_tag_age'] = {'type_info':{'0-17':1, '18-24':2, '25-34':3, '35-44':4, '45+':5}}
    smy['woeCal']['ft_gz_grey_list'] = {'type_info':{np.nan: 0, 'micro_loan_5_':2, 'micro_loan_3_4':1,
                                                     'micro_loan_5_,type_1':2, 'micro_loan_5_,type_2':2, 'micro_loan_3_4,type_1':1, 'micro_loan_3_4,type_2':1,
                                                     'type_1':0, 'type_2':0, 'type_1,type_2':0}}
    smy['woeCal']['ft_lbs_dis_label'] = {'type_info':{'d0':1, 'd1_300':2, 'd301_800':3, 'd801_2500':4, 'd2501_8000':5, 'd8001_20000':6, 'd20000_':7}}
    #js_smy = json.dumps(smy)
    tools.putFile(path+'/'+version, 'process_methods.json', smy)
    

if ifselect:
    #建议用非WOE编码的方式进行逻辑回归，查看模型效果
    #设定逻辑回归参数
    #保留test
    prc_methods = tools.getJson(path+'/'+version+'/process_methods.json')
    data, oot, data_lb, oot_lb = train_test_split(raw_data.drop('label', axis = 1), raw_data['label'], test_size = 0.2, random_state = rnd_seed)
    params= {'ifconst':True, 'ifnull':True}
    print('------------------------------------IV值计算---------------------------------------')
    ivBox = WoeFuncs(pct_size = 0.02, max_grps = 5, chiq_pv = 0.05, ifmono = True, keepnan = True, methods = 'tree')
    all_ivs = {}
    
    try:
        with tqdm(str_col+int_col+float_col) as t:
            for i in t:
                ivBox.setTgt(raw_data[[i, label]])
                if i in prc_methods['woeCal'].keys():
                    if isinstance(prc_methods['woeCal'][i]['type_info'], dict):
                        ivBox._setStrValue(prc_methods['woeCal'][i]['type_info'], ifraise = False)
                        ivBox.woe_cal()
                    elif prc_methods['woeCal'][i]['type_info'] == 'str':
                        ivBox.strWoe_cal()
                    else:
                        ivBox.woe_cal()
                elif i in str_col:
                    ivBox.strWoe_cal()
                else:
                    ivBox.woe_cal()
                    
                all_ivs[i] = ivBox.getIVinfo()
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    all_ivs = pd.DataFrame(pd.Series(all_ivs), columns = ['iv_value'])
    all_ivs = all_ivs.replace(np.inf, 0)
    all_ivs_detail = ivBox.woeDetail
    tools.putFile(path+'/'+version, 'ivsDetail.json', all_ivs_detail)
    print('------------------------------------woe编码---------------------------------------')
    pbox = AllFtrProcess(path+'/'+version+'/process_methods.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, keepnan = True, methods = 'tree')
    #全局样本上机型WOE分箱
    pbox = pbox.fit(raw_data)
    data_m = pbox.transform(data, iflabel = False)
    oot_m = pbox.transform(oot, iflabel = False)
    
    corr = data_m.corr()
    
if fittingpart:
    print('------------------------------------特征选择---------------------------------------')
    corr_ftr = 0.6
    ivs_ftr = 0.03
    size = 100
    all_models = []
    all_stat = []
    cv_params = {'params':params, 'modeltype':'lr', 'kfold':5}
    for i in [0.6, 0.7, 0.8]:
        for j in [0.02, 0.03, 0.04, 0.05, 0.06]:
            for size in [20,30,50,60,80,100,None]:
                sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, cv_params, path)
                #根据iv计算并选择特征
                prm_ftrs = sbox.ftr_filter(all_ivs[['iv_value']], size = size, tgt_c = j, corr_c = i)
                model = model_builder.cvModel(cv_params).fit(data_m[prm_ftrs], data_lb, test = oot_m[prm_ftrs], test_label = oot_lb)
                tvalues = model.getTvalues()
                pftr = model.Mperfrm
                all_stat += [['model_corr%s_ivs%s_size%s'%(i,j,str(size)),pftr['train']['auc'], pftr['train_std']['auc'], pftr['test']['auc'], pftr['test_std']['auc'], pftr['train']['ks'], pftr['train_std']['ks'], pftr['test']['ks'], pftr['test_std']['ks']]]
                
                tvalues.to_excel(path+'/'+version+'/tvalues'+'_model_corr%s_ivs%s_size%s'%(i,j,str(size))+'.xlsx')
                all_models = {'model_corr%s_ivs%s_size%s'%(i,j,str(size)):model}
                
    all_stat = pd.DataFrame(all_stat, columns = ['version', 'train_roc', 'train_roc_std', 'test_roc', 'test_roc_std', 'train_ks', 'train_ks_std', 'test_ks', 'test_ks_std'])
    all_stat.to_excel(path+'/'+version+'/model_stat.xlsx')
    
if oottest:
    corr_ftr = 0.7
    ivs_ftr = 0.04
    size = 80
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, cv_params, path)
    #根据iv计算并选择特征
    prm_ftrs = sbox.ftr_filter(all_ivs[['iv_value']], size = size, tgt_c = ivs_ftr, corr_c = corr_ftr)
    model = model_builder.cvModel(cv_params).fit(data_m[prm_ftrs], data_lb, test = None, test_label = None)
    tvalues = model.getTvalues()
    pftr = model.Mperfrm
    tvalues = pd.DataFrame(model.getTvalues(), columns = ['tvalues'])
    print(pd.DataFrame(model.Mperfrm))
#    tvalues.to_excel(path+'/'+version+'/tvalues.xlsx')
    data_pred = pd.DataFrame(pd.Series(model.predict(data_m[prm_ftrs]),index = data_m.index), columns = ['pred'])
    data_pred = data_pred.assign(label = data_lb)
    data_ks = FeatureStatTools.ks_cal_func(data_pred, grps=5, ascd = False, duplicates = 'drop')
    data_ks['bad_pct'].plot()
    oot_pred = pd.DataFrame(pd.Series(model.predict(oot_m[prm_ftrs]),index = oot_m.index), columns = ['pred'])
    oot_pred = oot_pred.assign(label = oot_lb)
    oot_ks = FeatureStatTools.ks_cal_func(oot_pred, grps=5, ascd = False, duplicates = 'drop')
    oot_ks['bad_pct'].plot()
    print('train的ks表现为：%s， oot的ks表现为：%s, 特征个数为：%s'%(data_ks['ks'].max(), oot_ks['ks'].max(), len(prm_ftrs)))
#
#    lftrs = list(set(list(data_m.columns.values))-set(prm_ftrs))
#    sbox.featureStat_model(prm_ftrs, modeltype = 'lr', rnd_seed = 21)
#    base = sbox.model_perform_['train']['auc']
#    
#    #控制50维特征为最，或者提升的roc不高于0.01
#    icr = base
#    while len(prm_ftrs) < 50 and icr > 0.01:
#        rlts = sbox.modelIprv_oneStep_plus(prm_ftrs, lftrs, modeltype = 'lr', rnd_seed = 21, mtrc = 'auc', eval_s = 'train')
#        tgt_ftr = list(rlts.keys())[0]
#        icr = rlts[tgt_ftr] - base
#        base = rlts[tgt_ftr]
#        if icr > 0.01:
#            prm_ftrs += [tgt_ftr]
#            lftrs.remove(tgt_ftr)

