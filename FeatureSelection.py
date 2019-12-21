import pandas as pd
import numpy as np

import re
import json
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import xgboost as xgb
import random
import warnings
from sklearn import metrics

import model_builder 
import tools
from WoeMethods import bins_method_funcs

import sklearn
if sklearn.__version__ > '0.20.0':
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split


class ModelBasedMethods(object):
    """
    docstring for TreeBasedMethods:
    提供基于模型的feature选择函数
    随机选择feature并迭代，并根据模型的表现确定特征
    随机样本
    """
    def __init__(self, ftrs, label, features, corr, params, path):
        self.ftrs = ftrs
        self.label = label
        if ftrs is not None:
            if len(set(list(features))-set(list(self.ftrs.columns.values))) >= 1:
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

    def _random_select(self, ftrs, musthave = None, nums = 20, rnd_seed = None):
        #随机挑选特征
        if rnd_seed is not None:
            random.seed(rnd_seed)

        if musthave is not None:
            for i in musthave:
                ftrs.remove(i)

        tmp = random.sample(ftrs, nums)
        if len(tmp) < nums:
            warnings.warn('not enough features for random feature selections')
        return tmp

    def _random_select_cor(self, ftrs, nums, musthave = None, corr_c = 0.75, rnd_seed = None):
        #随机挑选特征，但要考虑相关性
        corr = self.corr.copy()
        if rnd_seed is not None:
            random.seed(rnd_seed)

        if musthave is not None:
            rlts = musthave
        else:
            rlts = []
        pcr = list(corr.index.values)

        for i in rlts:
            ops = corr.loc[pcr,i]
            pcr = list(ops[ops<corr_c].index.values)

        while  len(rlts)<nums and len(pcr)>0:
            rlts += [random.sample(pcr, 1)[0]]
            pc = corr.loc[pcr, rlts[-1]]
            pc = pc[pc<corr_c]
            pcr = list(pc.index.values)

        if len(rlts) < nums:
            warnings.warn('not enough features for random feature selections')
        return rlts

    def _model_perform_funcs(self, ylabel, ypred):
        #统计模型表现
        rlt = {}
        rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
        rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
        rlt['logloss'] = metrics.log_loss(ylabel, ypred)
        return rlt

    def ftr_filter(self, tgt, tgt_c = 0.02, corr_c = 0.75):
        """
        根据特定指标及相关性进行特征选择
        tgt.shape = (n, 1)
        """
        corr = self.corr.copy()
        score_name = list(tgt.columns.values)[0]
        tgt.sort_values(score_name, ascending = False, inplace = True)
        tgt = tgt[score_name]
        tgt = tgt[tgt>tgt_c]
        lstd_bs = list(tgt.index.values)
        vld = []
        while len(lstd_bs) >= 1:
            vld += [lstd_bs[0]]
            lstd_bs = lstd_bs[1:]
            corr_check = corr[vld[-1]].loc[lstd_bs]
            corr_check = corr_check[corr_check>corr_c]
            invld = list(corr_check.index.values)

            for i in invld:
                lstd_bs.remove(i)

        return vld

    def _vif_filter(self, X, thres=10.0):
        """
        每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量
        理论上应该放到逻辑回归前面，暂时没有启用
        """
        col = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:,col].values, ix)
                   for ix in range(X.iloc[:,col].shape[1])]

            maxvif = max(vif)
            maxix = vif.index(maxvif)
            if maxvif > thres:
                del col[maxix]
                print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
                dropped = True
        print('Remain Variables:', list(X.columns[col]))
        print('VIF:', vif)
        return list(X.columns[col])

    def featureStat_model(self, ftrs = None, modeltype = 'lr', rnd_seed = None, test_size = 0.3):
        """
        根据model model_builder中的模型进行特征与模型表现的统计
        共有lrModel, xgbModel, cvModel三类
        """
        if ftrs is None:
            ftrs = self.features

        x = self.ftrs[ftrs]
        y = self.label
        if test_size == 0:
            train = x
            train_lb = y
            test = None
            test_lb = None
        train, test, train_lb, test_lb = train_test_split(x, y, test_size = test_size, random_state = rnd_seed)
        #return train, test, train_lb, test_lb 
        if modeltype == 'lr':
            if x.isnull().max().max() == 1:
                raise ValueError('unprocessed features with None values, pls check')

        if modeltype == 'lr':
            model = model_builder.lrModel(params = self.params)
        elif modeltype == 'xgb':
            model = model_builder.xgbModel(params = self.params)
        elif modeltype == 'cv':
            model = model_builder.cvModel(params = self.params)
        else:
            raise ValueError('other methods not provided')

        self.model_ = model.fit(train, train_lb, test, test_lb)
        self.model_perform_ = self.model_.getMperfrm()

    def getTvalues(self, mtrc, name = None):
        feat_imp = pd.DataFrame(self.model_.getTvalues(mtrc), columns = ['fscore'])
        stat = feat_imp.to_dict()
        lfts = [a for a in self.model_.ft_names if a not in stat['fscore'].keys()]
        for i in lfts:
            stat['fscore'][i] = 0

        if name is not None:
            json_str = json.dumps(stat, indent= 4, ensure_ascii= False)
            with open(self.path + '/feat_imps/'+name+'_ftimp.json', 'w', encoding= 'utf-8') as f:
                f.write(json_str)
                f.close()

            with open(self.path + '/feat_imps/all_auc.json', 'a') as f:
                f.write(name+'_ftimp %.4f %.4f\n'%(self.model_perform_['train']['auc'], self.model_perform_['test']['auc']))
                f.close()

        return pd.DataFrame(stat)

    def modelIprv_oneStep_plus(self, base_ftrs, tgts, modeltype = 'lr', rnd_seed = None, mtrc = 'auc', eval_s = 'train'):
        """
        可以用多进程优化
        """
        all_rlts = []
        for i in tgts:
            print(i)
            try:
                self.featureStat_model(base_ftrs+[i], modeltype, rnd_seed)
                all_rlts += [self.model_perform_[eval_s][mtrc]]
            except np.linalg.LinAlgError:
                all_rlts += [0]

        return {tgts[all_rlts.index(max(all_rlts))]:max(all_rlts)}

    def modelIprv_oneStep_minus(self, base_ftrs, modeltype = 'lr', rnd_seed = None, mtrc = 'auc', eval_s = 'train'):
        """
        可以用多进程优化
        """
        all_rlts = []
        for i in base_ftrs:
            ops = base_ftrs[:]
            ops.remove(i)
            try:
                self.featureStat_model(ops, modeltype, rnd_seed)
                all_rlts += [self.model_perform_[eval_s][mtrc]]
            except np.linalg.LinAlgError:
                all_rlts += [0]

        return {base_ftrs[all_rlts.index(max(all_rlts))]:max(all_rlts)}

    def featureAvgScore(self, top = None, ftr_c = 0.65):
        model_p = pd.read_table(self.path+'/feat_imps/all_auc.json', sep = ' ', names = ['files', 'train_auc', 'test_auc'])
        model_p = model_p[model_p['train_auc']>=ftr_c]
        vald_records = list(model_p['files'])

        for f in range(len(vald_records)):
            tgt = tools.getJson(self.path+'/feat_imps/'+vald_records[f]+'.json')
            rcd = pd.DataFrame(tgt)
            rcd.columns = [vald_records[f]]
            rcd.sort_values(vald_records[f], ascending = False, inplace = True)

            if top is not None:
                rcd = rcd.iloc[:top]
            if f == 0:
                rlt = rcd
            else:
                rlt = pd.merge(left = rlt, right = rcd, left_index = True, right_index = True, how = 'outer')
        
        if len(vald_records) == 0:
            raise ValueError('Not Enough Model')
        #return rlt
        fnl = pd.DataFrame(rlt.mean(axis = 1, skipna = True), columns = ['avg'])

        fnl['cnt'] = rlt.count(axis = 1, numeric_only = True)
        fnl['score'] = fnl['avg'] * fnl['cnt'].apply(lambda x: np.log(2+x))

        return fnl
