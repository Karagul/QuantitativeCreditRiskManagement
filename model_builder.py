#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:34:49 2019

@author: mobstaz_sc
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

import statsmodels.api as sm

import FeatureStatTools as funcs

#num_rounds = 500

class xgbModel(BaseEstimator, ClassifierMixin):
    """
    模型类比较相似，初始化之后会有都函数包括：
    1.进行参数调整都函数：setParams_;
    2.进行模型表现计算的函数：_model_perform_funcs;
    3.传入train，test并进行拟合的函数：fit；
    4.用以应用模型预测分数的函数：predict；
    5.用来获取特征重要性的函数：getTvalues;
    6.用来获取模型表现的函数:getMperfrm
    """
    def __init__(self, params = {'params':{'boosting':'gbtree','objective':'binary:logistic',
                    "nthread":20, 'eta':0.1, 'eval_metric': 'logloss',
                    'max_depth':5,'subsample':0.7,'colsample_bytree':0.7, 'gamma':0.2},
                    'early_stopping_rounds':10, 'num_rounds':200}):

        self.params = params['params']
        self.early_stopping_rounds = params['early_stopping_rounds']
        self.num_rounds = params['num_rounds']

    def setParams_(self, params):
        self.params = params

    def _model_perform_funcs(self, ylabel, ypred):
        rlt = {}
        if ylabel is None or ypred is None:
            rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
            rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
            rlt['logloss'] = metrics.log_loss(ylabel, ypred)
        else:
            rlt['auc'] = None
            rlt['apr'] = None
            rlt['logloss'] = None
        return rlt

    def fit(self, train, train_label, test, test_label, train_weight = None, test_weight = None):
        train_data = xgb.DMatrix(train, train_label, weight= train_weight)
        test_data = xgb.DMatrix(test, test_label, weight= test_weight)
        eval_watch = [(train_data, 'train'), (test_data, 'eval')]
        self.ft_names = list(train.columns.values)

        xgb_bst=xgb.train({**self.params},train_data, self.num_rounds,\
                          evals=eval_watch, early_stopping_rounds=self.early_stopping_rounds,verbose_eval=False)
        self.model_ = xgb_bst
        self.Mperfrm = {'train':self._model_perform_funcs(xgb_bst.predict(train), train_label), 'test':self._model_perform_funcs(xgb_bst.predict(test), test_label)}

        return self

    def predict(self, x, y = None):
        x = xgb.DMatrix(x, y)
        return np.array(self.model_.predict(x))

    def getTvalues(self, mtrc):
        return self.model_.get_fscore(mtrc)

    def getMperfrm(self):
        return self.Mperfrm

class lrModel(BaseEstimator, ClassifierMixin):
    """docstring for ."""
    def __init__(self, params= {'ifconst':True, 'ifnull':True}):
        self.ifconst = params['ifconst']
        self.ifnull = params['ifnull']

    def setParams_(self, params):
        self.params = params

    def _model_perform_funcs(self, ylabel, ypred):
        rlt = {}
        rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
        rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
        rlt['logloss'] = metrics.log_loss(ylabel, ypred)
        return rlt

    def fit(self, train, train_label, test = None, test_label = None, train_weight = None, test_weight = None):
        #self.train = train, self.train_label = train_label, self.test = test, self.test_label = test_label
        self.ft_names = list(train.columns.values)

        if self.ifconst:
            x = sm.add_constant(train)
        else:
            x = self.x.copy()

        if self.ifnull:
            x = x.fillna(0)

        model = sm.Logit(train_label, x).fit()
        self.model_ = model
        self.Mperfrm = {'train':self._model_perform_funcs(model.predict(train), train_label), 'test':self._model_perform_funcs(model.predict(test), test_label)}

        return self

    def predict(self, x, y = None):
        return np.array(self.model_.predict(x))

    def getTvalues(self, mtc = None):
        return self.model_.tvalues

    def getCoefs(self):
        return self.model_.params

    def getMperfrm(self):
        return self.Mperfrm

class cvModel(BaseEstimator, ClassifierMixin):
    def __init__(self, params = {'modeltype':'lr', 'kfold':5, 'params':{}}):
        if params['modeltype'] == 'lr':
            model = lrModel(params['params'])
        elif params['modeltype'] == 'xgb':
            model = xgbModel(params['params'])
        else:
            raise ValueError('unsupported methods')

        self.model_ = model
        self.kfolds = params['kfold']
        self.params = params['params']

    def setParams_(self, params):
        self.params = params

    def fit(self, train, train_label, test = None, test_label = None, train_weight = None, test_weight = None):
        kfolds = KFold(n_splits=self.kfolds,shuffle=False).split(train)
        models = []
        for train_index, test_index in kfolds:
            sub_train = train.loc[train_index]; sub_test = train.loc[test_index]
            sub_train_label = train_label.loc[train_index]; sub_test_label = train_label.loc[test_index]
            sub_train_weight = train_weight.loc[train_index]; sub_test_weight = train_weight.loc[test_index]

            models += [self.model_.fit(sub_train, sub_train_label, sub_test, sub_test_label, sub_train_weight, sub_test_weight)]

        self.models = models
        return self

    def predict(self, x, y = None):
        """
        this function returns the average prediction of all saved models!
        could be same bias when used to predict trainning sample
        """
        warnings.warn("""this function returns the average prediction of all functions!
                         could be same bias when used to predict trainning samples""")
        for m in range(len(self.models)):
            if m == 0:
                df = pd.DataFrame(self.models[m].predict(x,y), columns = ['model_'+str(m)])
            else:
                tmp = pd.DataFrame(self.models[m].predict(x,y), columns = ['model_'+str(m)])
                df = pd.merge(left = df, right = tmp, left_index = True, right_index = True, how = 'outer')


        return df.mean(axis = 1)

    def getTvalues(self, mtc = None):
        for m in range(len(self.models)):
            if m == 0:
                df = pd.DataFrame(self.models[m].getTvalues(mtc), columns = ['model_'+str(m)])
            else:
                tmp = pd.DataFrame(self.models[m].getTvalues(mtc), columns = ['model_'+str(m)])
                df = pd.merge(left = df, right = tmp, left_index = True, right_index = True, how = 'outer')

        df = df.fillna(0)
        return df.mean(axis = 1)

    def getMperfrm(self, train = None, train_label = None, test = None, test_label = None):
        rlts = [m.getMperfrm() for m in self.models]

        train = pd.DataFrame([a['train'] for a in rlts])
        test = pd.DataFrame([a['test'] for a in rlts])

        return {'train_mean':train.mean().to_dict(), 'test_mean':test.mean().to_dict(), 'train_std':train.std().to_dict(), 'test_std':test.std().to_dict()}
