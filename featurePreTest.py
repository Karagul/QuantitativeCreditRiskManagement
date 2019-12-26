import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import re
import json
from scipy import stats
import statsmodels.api as sm
import xgboost as xgb
import random
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold

import model_builder
import tools
from WoeMethods import bins_method_funcs
from FeatureProcess import AllFtrProcess
from FeatureSelection import ModelBasedMethods

#设置最基本的路径变量
level1 = False
level2 = False
level3 = False
level4 = True
level5 = False
path = '.'
woe_vrs = 'vrs1'
raw_data_file_name = 'data.csv'
path = '../function_test/raw_data'
rnd_seed = 21

type_check = tools.getJson(path + '/typ_info.json')
raw_data = pd.read_csv(path+'/'+raw_data_file_name, sep = r'[\t,|]', header = 0, dtype = {i:type_check[i]['type'] for i in type_check.keys() if type_check[i]['type'] == 'str'})
"""
提取之前计算的IV值，为后续评估做准备
"""
ivs = tools.getJson(path+'/feature_stat/ivStat.json')
ivs = pd.DataFrame(ivs).T
ivs['avg'] = ivs.mean(axis = 1)
"""
读取特征相关的统计
"""
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

raw_data = raw_data.drop(toDrop, axis = 1)

#
if level1:
    #建议用非WOE编码的方式进行逻辑回归，查看模型效果
    #设定逻辑回归参数
    #保留test
    data, oot, data_lb, oot_lb = train_test_split(raw_data.drop('label', axis = 1), raw_data['label'], test_size = 0.2, random_state = rnd_seed)
    params= {'ifconst':True, 'ifnull':True}

    pbox = AllFtrProcess(path+'/feature_process_methods/smy_level1.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
    pbox = pbox.fit(data.assign(label = data_lb))
    data_m = pbox.transform(data, iflabel = False)
    oot_m = pbox.transform(oot, iflabel = False)
    
    corr = data_m.corr()
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, params, path)
    #根据iv计算选择特征
    prm_ftrs = sbox.ftr_filter(ivs[['avg']], tgt_c = 0.03, corr_c = 0.6)

    lftrs = list(set(list(data_m.columns.values))-set(prm_ftrs))
    sbox.featureStat_model(prm_ftrs, modeltype = 'lr', rnd_seed = 21)
    base = sbox.model_perform_['train']['auc']
    
    #控制50维特征为最，或者提升的roc不高于0.01
    icr = base
    while len(prm_ftrs) < 50 and icr > 0.01:
        rlts = sbox.modelIprv_oneStep_plus(prm_ftrs, lftrs, modeltype = 'lr', rnd_seed = 21, mtrc = 'auc', eval_s = 'train')
        tgt_ftr = list(rlts.keys())[0]
        icr = rlts[tgt_ftr] - base
        base = rlts[tgt_ftr]
        if icr > 0.01:
            prm_ftrs += [tgt_ftr]
            lftrs.remove(tgt_ftr)

if level2:
    #建议用WOE编码的方式进行逻辑回归，查看模型效果
    #设定逻辑回归参数
    #level2与level1唯一不同在于特征的处理方式
    #保留test
    params= {'ifconst':True, 'ifnull':True}
    data, oot, data_lb, oot_lb = train_test_split(raw_data.drop('label' ,axis = 1), raw_data['label'], test_size = 0.2, random_state = rnd_seed)

    pbox = AllFtrProcess(path+'/feature_process_methods/smy_level1.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
    pbox = pbox.fit(data.assign(label = data_lb))
    data_m = pbox.transform(data, iflabel = False)
    oot_m = pbox.transform(oot, iflabel = False)
    
    corr = data_m.corr()
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, params, path)
    #根据iv计算选择特征
    prm_ftrs = sbox.ftr_filter(ivs[['avg']], tgt_c = 0.03, corr_c = 0.6)
    
    lftrs = list(set(list(data_m.columns.values))-set(prm_ftrs))
    sbox.featureStat_model(prm_ftrs, modeltype = 'lr', rnd_seed = 21)
    base = sbox.model_perform_['train']['auc']

    #控制50维特征为最，或者提升的roc不高于0.01
    while len(prm_ftrs) > 30 and icr >-0.01:
        #通过累加的方式判断是否需要添加特征
        rlts = sbox.modelIprv_oneStep_minus(prm_ftrs, modeltype = 'lr', rnd_seed = 21, mtrc = 'auc', eval_s = 'train')
        tgt_ftr = list(rlts.keys())[0]
        icr = rlts[tgt_ftr] - base
        base = rlts[tgt_ftr]
        if icr > -0.01:
            prm_ftrs.remove(tgt_ftr)
            
        

if level3:
    #使用xgboost
    #feature一般不做特别处理，除非cap
    #通过随机抽取特征的方式计算
    #需要留出OOT
    model_params = {
    #'booster':'gbtree',
    'objective': 'binary:logistic', #多分类的问题
    #'eval_metric': 'auc',
    #'num_class':10, # 类别数，与 multisoftmax 并用
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':3, # 构建树的深度，越大越容易过拟合
    'lambda':1000,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight': 5,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1, # 如同学习率
    'seed':1000,
    'nthread':-1,# cpu 线程数
    'eval_metric': 'logloss'
    }

    params = {'params':model_params, 'early_stopping_rounds':10, 'num_rounds':50}
    data, oot, data_lb, oot_lb = train_test_split(raw_data.drop('label' ,axis = 1), raw_data['label'], test_size = 0.2, random_state = rnd_seed)

    pbox = AllFtrProcess(path+'/feature_process_methods/smy_level1.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
    pbox = pbox.fit(data.assign(label = data_lb))
    data_m = pbox.transform(data, iflabel = False)
    oot_m = pbox.transform(oot, iflabel = False)

    all_ftrs = list(data_m.columns.values)
    corr = data_m.corr()
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, params, path)

    for i in range(10):
        #随机抽取10000次样本并建模
        #this takes time, do remember~
        features = sbox._random_select_cor(all_ftrs, 50, musthave = None, corr_c = 0.75, rnd_seed = None)
        sbox.featureStat_model(features, modeltype = 'xgb', rnd_seed = 21)
        _ = sbox.getTvalues('gain', ifabs = True, name = 'grp_'+str(i))

    rlts = sbox.featureSelection_AvgScore(top = 30, ftr_c = 0.55)
    rlts.sort_values('score', ascending =False, inplace = True)
    prm_ftrs = list(rlts.index.values)[:50]

if level4:
    #维信方法
    #随机筛选的方法，主要用gain进行评估
    #使用xgboost
    #feature一般不做特别处理，除非cap
    #通过随机抽取特征的方式计算
    #需要留出OOT
    model_params = {
    #'booster':'gbtree',
    'objective': 'binary:logistic', #多分类的问题
    #'eval_metric': 'auc',
    #'num_class':10, # 类别数，与 multisoftmax 并用
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':3, # 构建树的深度，越大越容易过拟合
    'lambda':1000,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight': 5,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1, # 如同学习率
    'seed':1000,
    'nthread':-1,# cpu 线程数
    'eval_metric': 'logloss'
    }

    params = {'params':model_params, 'early_stopping_rounds':10, 'num_rounds':50}
    data, oot, data_lb, oot_lb = train_test_split(raw_data.drop('label' ,axis = 1), raw_data['label'], test_size = 0.2, random_state = rnd_seed)

    pbox = AllFtrProcess(path+'/feature_process_methods/smy_level1.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
    pbox = pbox.fit(data.assign(label = data_lb))
    data_m = pbox.transform(data, iflabel = False)
    oot_m = pbox.transform(oot, iflabel = False)

    all_ftrs = list(data_m.columns.values)
    corr = data_m.corr()
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, params, path)

    features = sbox._random_select_cor(all_ftrs, 200, musthave = None, corr_c = 0.75, rnd_seed = None)
    prm_ftrs = sbox.featureSelection_randomSelect(ftr_names = features, modeltype = 'xgb', importance_type='gain',\
                                                  threshold1 = 0.02,threshold2=0.01, threshold3=5, keep_rate=0.5, \
                                                  max_iter=100, min_num = 5, test_size = 0.3)

if level5:
    #使用xgboost并且用cv的方式评估模型
    #feature一般不做特别处理，除非cap
    #通过随机抽取特征的方式计算
    #不需要留出OOT
    model_params = {
    #'booster':'gbtree',
    'objective': 'binary:logistic', #多分类的问题
    #'eval_metric': 'auc',
    #'num_class':10, # 类别数，与 multisoftmax 并用
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':3, # 构建树的深度，越大越容易过拟合
    'lambda':1000,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #'subsample':0.7, # 随机采样训练样本
    #'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight': 5,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1, # 如同学习率
    'seed':1000,
    'nthread':-1,# cpu 线程数
    'eval_metric': 'logloss'
    }

    xgb_params = {'params':model_params, 'early_stopping_rounds':10, 'num_rounds':50}
    params = {'modeltype':'xgb', 'params':xgb_params, 'kfold':5}

    pbox = AllFtrProcess(path+'/feature_process_methods/smy_level1.json',\
                         pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')
    data = raw_data.drop('label', axis = 1)
    data_lb = raw_data['label']
    pbox = pbox.fit(data.assign(label = data_lb))
    data_m = pbox.transform(data, iflabel = False)
    all_ftrs = list(data_m.columns.values)

    corr = data_m.corr()
    sbox = ModelBasedMethods(data_m, data_lb, list(data_m.columns.values), corr, params, path)

    for i in range(10):
        #随机抽取10000次样本并建模
        #this takes time, do remember~
        #可以采用多进程优化
        features = sbox._random_select_cor(all_ftrs, 50, musthave = None, corr_c = 0.75, rnd_seed = None)
        sbox.featureStat_model(features, modeltype = 'cv', rnd_seed = 21)
        _ = sbox.getTvalues('gain', ifabs = True, name = 'grp_'+str(i))

    rlts = sbox.featureSelection_AvgScore(top = 30, ftr_c = 0.5)
    rlts.sort_values('score', ascending =False, inplace = True)
    prm_ftrs = list(rlts.index.values)[:50]
