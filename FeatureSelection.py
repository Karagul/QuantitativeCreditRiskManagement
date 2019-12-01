import pandas as pd
import numpy as np

import re
import json
from scipy import stats
import xgboost as xgb
import random
import warnings

import model_builder
from tools import *
from WoeMethods import bins_method_funcs

import sklearn
if sklearn.__version__ > '0.20.0':
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split

params = {
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


class ModelBasedMethods(object):
    """
    docstring for TreeBasedMethods:
    提供基于模型的feature选择函数
    随机选择feature并迭代，并根据模型的表现确定特征
    随机样本
    """
    def __init__(self, raw, features, corr, params, path):
        self.raw = raw
        self.features = features
        if raw is not None:
            if len(set(list(features))-set(list(self.raw.columns.values))) >= 1:
                raise ValueError('feature N data not match!')
        self.corr = corr
        self.params = params
        self.path = path

    def setFeatures(self, ftrs):
        self.features = ftrs

    def setCorr(self, corr):
        self.corr = corr

    def setParams(self, params):
        try:
            for i in params.keys():
                self.params[i] = params[i]
        except:
            raise ValueError('Invalid Parameters')

    def _random_select(self, nums, rnd_seed = None):
        ftrs = self.features[:]
        if rnd_seed is not None:
            random.seed(rnd_seed)
        tmp = random.sample(ftrs, nums)
        if len(tmp) < nums:
            warnings.warn('not enough features for random feature selections')
        self.tgt_ftrs = tmp

    def _random_select_cor(self, ftrs, nums, corr_c = 0.75, rnd_seed = None):
        ftrs = self.features[:]
        corr = self.corr.copy()
        if rand_seed is not None:
            random.seed(rnd_seed)

        rlts = []

        while  len(rlts)<nums or len(ftrs)>0:
            rlts += [random.sample(ftrs, 1)[0]]
            pc = corr[rlts[-1]]
            pc = pc[pc<corr_c]
            ftrs = list(pc.index.values)

        if len(tmp) < nums:
            warnings.warn('not enough features for random feature selections')
        self.tgt_ftrs = tmp

    def _ftr_filter(self, tgt, tgt_c = 0.02, corr_c = 0.75):
        """
        根据IV值及相关性进行特征选择
        """
        corr = self.corr.copy()
        tgt.sort_values('iv', ascending = False, inplace = True)
        tgt = tgt[tgt['iv']]
        lstd_bs = list(tgt['ft_names'])
        vld = []
        while len(lstd_bs) >= 1:
            vld += [lstd_bs[0]]
            lstd_bs = lstd_bs[1:]
            corr_check = corr[i].loc[lstd_bs][corr[i]>corr_c]
            invld = list(corr_check.index.values)

            for i in invld:
                lstd_bs.remove(i)

        return vld

    def featureStat(self, name, rnd_seed = None, dir_output = False):
        x = self.raw[self.tgt_ftrs]
        y = self.raw['label']

        train, test, train_lb, test_lb = train_test_split(x, y, test_size = 0.3, random_state = rnd_seed)
        dtrain = xgb.DMatrix(train, label = train_lb, missing = np.nan)
        dtest  = xgb.DMatrix(test, label = test_lb, missing = np.nan)

        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        model = xgb.train(self.params, dtrain, rounds, watchlist, early_stopping_rounds = 200)
        feat_imp = pd.DataFrame(model.get_fscore(), index = ['fscore']).T

        stat = feat_imp['fscore'].to_dict()
        lfts = [a for a in self.tgt_ftrs if a not in stat.keys()]
        for i in lftrs:
            lfts[i] = 0
        json_str = json.dumps(psi_new, indent= 4, ensure_ascii= False)
        with open(path+'/feat_imps/'+name+'.json', 'w', encoding= 'utf-8') as f:
            f.write(json_str)
            f.close()

        if dir_output:
            return feat_imp

    def featureAvgScore(self, top = None):
        scores_records = tools.getFiles(self.path+'/feat_imps')
        for f in range(len(scores_records)):
            tgt = tools.getJson(self.path+'/feat_imps/'+scores_records[f])
            rcd = pd.DataFrame({scores_records[f]:tgt})
            rcd.sort_values(scores_records[f], ascending = False, inplace = True)

            if top is not None:
                rcd = rcd.iloc[:top]
            if f == 0:
                rlt = rcd
            else:
                rlt = pd.merge(left = rlt, right = rcd, left_index = True, right_index = False, how = 'outer')

        rlt['avg'] = rlt.mean(axis = 1, skipna = True)
        rlt['cnt'] = rlt.count(axis = 1, numerical_only = True)
        rlt['score'] = rlt['avg'] * rlt['cnt'].apply(lambda x: np.log(2+x))

        return rlt
