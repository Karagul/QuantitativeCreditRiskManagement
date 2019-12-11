import pandas as pd
import numpy as np

import re
import os
import json
import warnings

from sklearn import preprocessing
from scipy import stats

from WoeMethods import *


class FeatureProcessFuncs(AllWoeFuncs):
    """docstring fs usprvFeatureProcess.
    a specific dicionary, indicating processing methods for average single features is required here:
    1.No Processing;
    2.fillna with (mean, average, min, max, or any other value)；
    3.one hot;
    4.set quantitative values for str values;
    5.woe coding for (str or numbers)
    =======================================================
    setData heritated from AllWoeFuncs will be used for all data s
    """
    def __init__(self, path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
        super(FeatureProcessFuncs, self).__init__(path, pct_size, max_grps, chiq_pv, ifmono, ifnan, methods)

    def setTgt(self, df):
        self.df = df
        self.ft_name = list(df.columns.values)[0]

    def fillmethods(self, value, inplace = False):
        if isinstance(value, int) or isinstance(value, float):
            tgt_v = value
        elif value in ['mean', 'median', 'min', 'max']:
            tgt_v = eval('self.df[self.ft_name].%s()'%value)
        else:
            tgt_v = value
            warnings.warn('str value input for missing fulfill, please check!')

        if inplace:
            self.df = self.df.fillna(tgt_v)
        else:
            self.mdf = self.df.fillna(tgt_v)

    def _subcapMethods(self, x, s, b):
        if s is None and b is None:
            return x
        elif s is None and b is not None:
            return min(x, b)
        elif s is not None and b is None:
            return max(x, s)
        else:
            return min(b, max(x, s))

    def capMethods(self, up = None, floor = None, max_pct = None, min_pct = None, inplace = False):
        if (up is None) and (max_pct is not None):
            up = self.df[self.ft_name].quantile(max_pct)

        if (floor is None) and (min_pct is not None):
            floor = self.df[self.ft_name].quantile(min_pct)

        data = self.df.copy()

        if (up is None) and (floor is None):
            warnings.warn('Function dose not actually run')
        data[self.ft_name] = data[self.ft_name].apply(self._subcapMethods, (floor, up))

        if inplace:
            self.df = data
        else:
            self.mdf = data

    def onehot_sf(self, data = None, value_range = None):
        ft_name = self.ft_name, data = self.df.copy()

        if value_range is not None:
            data[ft_name] = data['ft_name'].apply(lambda x: x if x in value_range else 'nan')

        if 'nan' in data[ft_name]:
            mdf = pd.get_dummies(data[ft_name])
            mdf.columns = [ft_name+'_'+str(a) for a in mdf.columns.values]
            mdf = mdf.drop(ft_name+'_nan')
        else:
            mdf = pd.get_dummies(data[ft_name], drop_first = True)
            mdf.columns = [ft_name+'_'+str(a) for a in mdf.columns.values]

        self.mdf = mdf

    def setStrMethods(self, strs2orders, inplace = False):
        """
        给予字符串变量可比较性；
        strs2orders : 一个字典包含对应的字符串与数值对应关系
        """
        data = self.df.copy()
        data[self.ft_name] = data[self.ft_name].apply(lambda x: strs2orders[x] if x in strs2orders.keys() else None)
        if inplace:
            self.df = data
        else:
            self.mdf = data

    def RobustScaler(data, apply_data = None, **args):
        """
        using the scaler method provided by preprocessing, params are followed:
        with_centering : boolean, True by default
            If True, center the data before scaling.
            This will cause ``transform`` to raise an exception when attempted on
            sparse matrices, because centering them entails building a dense
            matrix which in common use cases is likely to be too large to fit in
            memory.

        with_scaling : boolean, True by default
            If True, scale the data to interquartile range.

        quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
            Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
            Quantile range used to calculate ``scale_``.

        copy : boolean, optional, default is True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array or scipy.sparse CSR matrix, a copy may still be
            returned.
        """
        prs = preprocessing.RobustScaler(**args).fit(data)
        if apply_data is None:
            self.mdf = prs.transform(data)
        else:
            self.mdf = prs.transform(apply_data)

    def minmax_scaler(data, apply_data = None, **args):
        """
        another funcstions that can be directorly inheritated from preprocessing
        """
        pass

    def feature_egineer(self):
        """
        save for feature_egineer
        """
        pass

    def getMdf(self):
        return self.mdf

    def collectSummary(self, path):
        smy = getFile(path)
        self.undo_list = smy['undo']
        self.fill_list = smy['fill']
        self.cap_list = smy['cap'] #
        self.var2char_list = smy['var2char'] #should be a dictionary
        self.onehot_list = smy['onehot']
        self.woe_list = smy['woeCal']


class FeatureProcess(FeatureProcessFuncs):
    """docstring for ."""
    def __init__(self, arg):
        super(FeatureProcess, self).__init__(path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree')

    def setData(self, data):
        self.data = data

    def collectSummary(self, path):
        smy = getFile(path)
        self.undo_list = smy['undo']
        self.fill_list = smy['fill']
        self.cap_list = smy['cap'] #
        self.var2char_list = smy['var2char'] #should be a dictionary
        self.onehot_list = smy['onehot']
        self.woe_list = smy['woeCal']

    def allProcess(self, vrs, ifsave = True):
        """
        vrs:indicating the version of woe coding in this function
        """
        mdf = self.data[self.undo_list+['label']]
        """
        缺失值填充
        """
        for i in self.fill_list.keys():
            self.setTgt(self.data[[i]])
            self.fillmethods(self.fill_list[i], inplace = False)
            mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')

        """
        去除极值
        """
        for i in self.cap_list.keys():
            self.setTgt(self.data[i])
            up = self.cap_list[i].get('up'); floor = self.cap_list[i].get('floor')
            max_pct = self.cap_list[i].get('max_pct'); min_pct = self.cap_list[i].get('min_pct')
            self.capMethods(up, floor, max_pct, min_pct, inplace = False)
            mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')

        """
        处置字符串型变量
        """
        for i in self.var2char_list.keys():
            self.setTgt(self.data[i])
            self.setStrMethods(self.var2char_list[i], inplace = False)
            mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')

        """
        Onehot变量处理
        """
        for i in self.onehot_list.keys():
            self.setTgt(self.data[i])
            self.onehot_sf(self.onehot_list[i].get('value_range'))
            mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')

        """
        woe编码:precalculated iv info needed
        """
        self.setData(self.data[list(self.woe_list.keys())+['label']])
        self.setFtrs(self.woe_list)
        try:
            self.AllWoeCollects(vrs)
        except:
            warnings.warn('no given woe info detetd, all data will be used to calculate woe first')
            self.AllWocCals(vrs)
        self.AllWoeApl(ifsave = False)

        mdf = pd.merge(left = mdf, right = self.getMdfData().drop('label', axis = 1), left_index = True, right_index = True)

        if ifsave:
            mdf.to_csv(self.path+'/FeatureModify/DataToModel.csv')
        self.finalData = mdf

    def getFinalData(self):
        try:
            return self.finalData
        except:
            return None
