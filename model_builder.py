#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:34:49 2019

@author: mobstaz_sc
"""

import pandas as pd
import numpy as np

#import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib

import statsmodels.api as sm

import data_test as funcs

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

#num_rounds = 500

def lr_model(train, test, add_const = True):
    #all_d = pd.concat([train, test])
    #p_m = all_d['phone_price_price'].median()

    train_y = train['label']
    train_x = train.drop('label', axis = 1)
    test_y = test['label']
    test_x = test.drop('label', axis = 1)
    if 'phone_price_price' in train.columns.values:
        all_d = pd.concat([train, test])
        p_m = all_d['phone_price_price'].median()
        test_x['phone_price_price'] = test_x['phone_price_price'].fillna(p_m)
        train_x['phone_price_price'] = train_x['phone_price_price'].fillna(p_m)

    if add_const:
        test_x = sm.add_constant(test_x.fillna(0))
        train_x = sm.add_constant(train_x.fillna(0))
    else:
        test_x = test_x.fillna(0)
        train_x = train_x.fillna(0)
    model = sm.Logit(train_y, train_x).fit()

    xpred = model.predict(train_x)
    ypred = model.predict(test_x)

    ft_imp = pd.DataFrame(model.tvalues, columns = ['tvalue'])
    df = pd.DataFrame([ypred, test_y], index = ['pred', 'label']).T
    ks = funcs.ks_cal(df, grps = 10, ascd = False)

    df_train = pd.DataFrame([xpred, train_y], index = ['pred', 'label']).T
    ks_train = funcs.ks_cal(df_train, grps = 10, ascd = False)

    score_test = [metrics.roc_auc_score(test_y, ypred)]
    score_test += [metrics.average_precision_score(test_y, ypred)]
    score_test += [metrics.log_loss(test_y, ypred)]
    score_test += [ks['ks'].max()]

    score_train = [metrics.roc_auc_score(train_y, xpred)]
    score_train += [metrics.average_precision_score(train_y, xpred)]
    score_train += [metrics.log_loss(train_y, xpred)]
    score_train += [ks_train['ks'].max()]

    return ft_imp, score_test, score_train, df, df_train, ks


def xgb_model(train, test, params = params, rounds = 500):
    train_y = train['label']
    train_x = train.drop('label', axis = 1)
    test_y = test['label']
    test_x = test.drop('label', axis = 1)

    dtrain = xgb.DMatrix(train_x, label = train_y, missing = np.nan)
    dtest  = xgb.DMatrix(test_x, label = test_y, missing = np.nan)

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, rounds, watchlist, early_stopping_rounds = 200)
    ypred = model.predict(dtest)

    df = pd.DataFrame([ypred, test_y], index = ['pred', 'label']).T
    ks = funcs.ks_cal(df, grps = 10, ascd = False)

    df_train = pd.DataFrame([xpred, train_y], index = ['pred', 'label']).T
    ks_train = funcs.ks_cal(df_train, grps = 10, ascd = False)

    feat_imp = pd.DataFrame(model.get_fscore(), index = ['fscore']).T
    feat_imp.sort_values('fscore', inplace = True, ascending = False)
    feat_imp['fscore'] = feat_imp['fscore']/feat_imp['fscore'].sum()

    score_test = [metrics.roc_auc_score(test_y, ypred)]
    score_test += [metrics.average_precision_score(test_y, ypred)]
    score_test += [metrics.log_loss(test_y, ypred)]
    score_test += [ks['ks'].max()]

    score_train = [metrics.roc_auc_score(train_y, xpred)]
    score_train += [metrics.average_precision_score(train_y, xpred)]
    score_train += [metrics.log_loss(train_y, xpred)]
    score_train += [ks_train['ks'].max()]

    #y_pred = pd.DataFrame([test_y, ypred], columns = ['label', 'pred'])

    return feat_imp, score_test, score_train, df, df_train


def xgb_model_with_train(train, test, params = params, rounds = 500):
    train_y = train['label']
    train_x = train.drop('label', axis = 1)
    test_y = test['label']
    test_x = test.drop('label', axis = 1)

    dtrain = xgb.DMatrix(train_x, label = train_y, missing = np.nan)
    dtest  = xgb.DMatrix(test_x, label = test_y, missing = np.nan)

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, rounds, watchlist, early_stopping_rounds = 200)
    ypred = model.predict(dtest)
    xpred = model.predict(dtrain)

    df = pd.DataFrame([ypred, test_y], index = ['pred', 'label']).T
    ks = funcs.ks_cal(df, grps = 10, ascd = False)

    df_train = pd.DataFrame([xpred, train_y], index = ['pred', 'label']).T
    ks_train = funcs.ks_cal(df_train, grps = 10, ascd = False)

    feat_imp = pd.DataFrame(model.get_fscore(), index = ['fscore']).T
    feat_imp.sort_values('fscore', inplace = True, ascending = False)
    feat_imp['fscore'] = feat_imp['fscore']/feat_imp['fscore'].sum()

    score_test = [metrics.roc_auc_score(test_y, ypred)]
    score_test += [metrics.average_precision_score(test_y, ypred)]
    score_test += [metrics.log_loss(test_y, ypred)]
    score_test += [ks['ks'].max()]

    score_train = [metrics.roc_auc_score(train_y, xpred)]
    score_train += [metrics.average_precision_score(train_y, xpred)]
    score_train += [metrics.log_loss(train_y, xpred)]
    score_train += [ks_train['ks'].max()]

    return model, feat_imp, score_test, score_train, df, df_train, ks
