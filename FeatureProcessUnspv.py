import pandas as pd
import numpy as np

import re
import os
import json
import warnings

from sklearn import preprocessing
from scipy import stats


class FeatureProcessFuncs(object):
    """docstring fs usprvFeatureProcess."""
    def __init__(self):
        pass

    def setData(self, df):
        self.df = df
        self.ft_name = list(df.columns.values)[0]
        self.basic_stat = {}
        self.basic_stat['mean'] = df[self.ft_name].mean()
        self.basic_stat['median'] = df[self.ft_name].median()
        self.basic_stat['min'] = df[self.ft_name].min()
        self.basic_stat['max'] = df[self.ft_name].max()

    def fillmethods(self, value, data = None):
        if isinstance(value, int):
            tgt_v = value
        else:
            try:
                tgt_v = self.basic_stat[value]
            except:
                tgt_v = value
                warnings.warn('str value input for missing fulfill, please check!')

        if data is None:
            self.mdf = self.df.fillna(tgt_v)
        else:
            self.mdf = data.fillna(tgt_v)

    def onehot_sf(data, value_range = None):
        ft_name = data.columns.values[0]
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

    def getMdf(self):
        return self.mdf


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
