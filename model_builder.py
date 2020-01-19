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
from sklearn.tree import DecisionTreeClassifier

import statsmodels.api as sm
from copy import copy

import FeatureStatTools as funcs
import warnings

from FeatureStatTools import ks_cal_func


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

    def setParams(self, params):
        try:
            for i in params.keys():
                self.params[i] = params[i]
        except:
            raise ValueError('Invalid Parameters')

    def _model_perform_funcs(self, ylabel, ypred):
        rlt = {}
        if ylabel is not None and ypred is not None:
            rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
            rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
            rlt['logloss'] = metrics.log_loss(ylabel, ypred)
            data_pred = pd.DataFrame(pd.Series(ypred,index = ylabel.index), columns = ['pred'])
            data_pred = data_pred.assign(label = ylabel)
            try:
                rlt['ks'] = ks_cal_func(data_pred, grps=5, ascd = False, duplicates = 'drop')['ks'].max()
            except:
                rlt['ks'] = None
        else:
            rlt['auc'] = None
            rlt['apr'] = None
            rlt['logloss'] = None
            rlt['ks'] = None
        return rlt

    def fit(self, train, train_label, test, test_label, train_weight = None, test_weight = None):
        train_data = xgb.DMatrix(train, train_label, weight= train_weight)
        test_data = xgb.DMatrix(test, test_label, weight= test_weight)
        eval_watch = [(train_data, 'train'), (test_data, 'eval')]
        self.ft_names = list(train.columns.values)

        xgb_bst=xgb.train({**self.params},train_data, self.num_rounds,\
                          evals=eval_watch, early_stopping_rounds=self.early_stopping_rounds,verbose_eval=False)
        
        train_pred = xgb_bst.predict(train_data)
        if test is not None:
            test_pred = xgb_bst.predict(test_data)
        else:
            test_pred = None
        self.model_ = xgb_bst
        self.Mperfrm = {'train':self._model_perform_funcs(train_label,train_pred), 'test':self._model_perform_funcs(test_label,test_pred)}
        
        return self

    def predict(self, x, y = None):
        x = xgb.DMatrix(x, y)
        return np.array(self.model_.predict(x))

    def getTvalues(self, mtrc):
        return pd.Series(self.model_.get_score(importance_type=mtrc))

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
        if ylabel is not None and ypred is not None:
            rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
            rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
            rlt['logloss'] = metrics.log_loss(ylabel, ypred)
            data_pred = pd.DataFrame(pd.Series(ypred,index = ylabel.index), columns = ['pred'])
            data_pred = data_pred.assign(label = ylabel)
            try:
                rlt['ks'] = ks_cal_func(data_pred, grps=5, ascd = False, duplicates = 'drop')['ks'].max()
            except:
                rlt['ks'] = None
        else:
            rlt['auc'] = None
            rlt['apr'] = None
            rlt['logloss'] = None
            rlt['ks'] = None
        return rlt

    def fit(self, train, train_label, test = None, test_label = None, train_weight = None, test_weight = None):
        #self.train = train, self.train_label = train_label, self.test = test, self.test_label = test_label
        self.ft_names = list(train.columns.values)

        if self.ifconst:
            x = sm.add_constant(train)
            if test is not None:
                x_test = sm.add_constant(test)
        else:
            x = train
            if test is not None:
                x_test = test

        if x.isna().sum().sum()>0:
            warnings.warn('exist na data in logistic regression fitting')
            x = x.fillna(0)
            train=train.fillna(0)
            if not test is None:
                test=test.fillna(0)

        model = sm.Logit(train_label, x).fit()
        self.model_ = model
        train_pred = model.predict(x)
        if test is not None:
            test_pred = model.predict(x_test)
        else:
            test_pred = None
        self.Mperfrm = {'train':self._model_perform_funcs(train_label,train_pred), 'test':self._model_perform_funcs(test_label,test_pred)}

        return self

    def predict(self, x, y = None):
        if self.ifconst:
            x = sm.add_constant(x)
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

    def setParams(self, params):
        try:
            for i in params.keys():
                self.params[i] = params[i]
        except:
            raise ValueError('Invalid Parameters')

    def fit(self, train, train_label, test = None, test_label = None, train_weight = None, test_weight = None):
        kfolds = KFold(n_splits=self.kfolds,shuffle=True).split(train)
        models = []
        for train_index, test_index in kfolds:
            sub_train = train.iloc[train_index]; sub_test = train.iloc[test_index]
            sub_train_label = train_label.iloc[train_index]; sub_test_label = train_label.iloc[test_index]
            if train_weight is not None:
                sub_train_weight = train_weight.iloc[train_index]; sub_test_weight = train_weight.iloc[test_index]
            else:
                sub_train_weight = None; sub_test_weight = None

            models += [copy(self.model_.fit(sub_train, sub_train_label, sub_test, sub_test_label, sub_train_weight, sub_test_weight))]

        self.models = models
        self.ft_names = list(train.columns.values)
        self.Mperfrm = self.getMperfrm()
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


        return np.array(df.mean(axis=1))

    def getTvalues(self, mtc = None):
        for m in range(len(self.models)):
            if m == 0:
                df = pd.DataFrame(self.models[m].getTvalues(mtc), columns = ['model_'+str(m)])
            else:
                tmp = pd.DataFrame(self.models[m].getTvalues(mtc), columns = ['model_'+str(m)])
                df = pd.merge(left = df, right = tmp, left_index = True, right_index = True, how = 'outer')

        df = df.fillna(0)
        return df
        #return df.mean(axis = 1)

    def getMperfrm(self, train = None, train_label = None, test = None, test_label = None):
        rlts = [m.getMperfrm() for m in self.models]

        train = pd.DataFrame([a['train'] for a in rlts])
        test = pd.DataFrame([a['test'] for a in rlts])

        return {'train':train.mean().to_dict(), 'test':test.mean().to_dict(), 'train_std':train.std().to_dict(), 'test_std':test.std().to_dict()}

class trAdaboostMethods(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 100, max_depth = 2, min_samples_split = 2, random_state = None):
        self.n_estimators = n_estimators
        self.max_depth = 2
        self.min_samples_split = 2
        self.random_state = random_state
    
    def _calculateP(self, weights):
        total = weights.sum()
        return weights / total
    
    def _estimator(self, trans_data, trans_label, test_data, p):
        clf = DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random", \
                                     max_depth = self.max_depth, min_samples_split = self.min_samples_split, random_state = self.random_state)
        clf.fit(trans_data, trans_label, sample_weight=p)
        return clf.predict(test_data)
    
    def _errorRate(self, label_R, label_H, weight):
        total = np.sum(weight)
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))
    
    def _smlSum(self, alist):
        
        rlts = alist[0]
        for i in alist[1:]:
            rlts += alist[i]
            
        return rlts
    
    def _model_perform_funcs(self, ylabel, ypred):
        rlt = {}
        if ylabel is not None and ypred is not None:
            rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
            rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
            rlt['logloss'] = metrics.log_loss(ylabel, ypred)
        else:
            rlt['auc'] = None
            rlt['apr'] = None
            rlt['logloss'] = None
        return rlt
    
    def fit(self, train, train_label, test = None, test_label = None):
        """
        a specific indicator is required suggesting sample distribution
        """
        h_list = []
        beta_ts = []
        s_index = train[train['if_same_dist'] == 1].index
        d_index = train[train['if_same_dist'] == 0].index
        train = train.drop('if_same_dist', axis = 1)
        
        weights = pd.Series(np.ones([len(train)])/len(train), index = train.index)
        beta = 1/(1+np.sqrt(2*np.log(len(d_index)/self.n_estimators)))
        
        for i in range(self.n_estimators):
            p = self._calculateP(weights)
            
            clf = DecisionTreeClassifier(max_depth = self.max_depth, min_samples_split = self.min_samples_split, random_state = self.random_state)
            clf.fit(train, train_label, sample_weight=p)
            pred = pd.Series(clf.predict(train), idnex = train.index)
            
            es = self._errorRate(pred.loc[s_index], train_label.loc[s_index], weights.loc[s_index])
            if es > 0.5:
                warnings.warn('error rates too high, may not converge!')
                es = 0.5
            
            beta_t = es/(1-es)
                
            w_s = weights.loc[s_index] * (pred-train_label).loc[s_index].apply(np.abs).apply(lambda x: np.power(beta_t, (-x)))
            w_d = weights.loc[d_index] * (pred-train_label).loc[d_index].apply(np.abs).apply(lambda x: np.power(beta,x))
            weights = pd.concat([w_s, w_d])
            h_list += [clf]
            beta_ts += [beta_t]
            
        self.models = h_list
        self.beta_ts = beta_ts
        
        return self
    
    def predict(self, x, y = None):
        pred = self._smlSum([-self.models[i].predict(x) * np.log(self.beta_ts[i]) for i in range(np.int(np.ceil(self.n_estimators/2)), self.n_estimators+1)])
        ctrs = np.sum([-0.5 * np.log(self.beta_ts[i]) for i in range(np.int(np.ceil(self.n_estimators/2)), self.n_estimators+1)])
            
        rlts = (pred>=ctrs).apply(np.int)
        
        return rlts
    
    def getTvalues(self, mtc = None):
        pass

    def getCoefs(self):
        pass

    def getMperfrm(self):
        pass
            
            
                
            
                
        
        
        
        
    
    