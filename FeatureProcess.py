import pandas as pd
import numpy as np

import re
import os
import json
import warnings
from tqdm import tqdm

from sklearn import preprocessing
from scipy import stats

from WoeMethods import AllWoeFuncs, WoeFuncs
import tools

class undoMethods(object):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df
    
    

class fillMethods(object):
    def __init__(self, param):
        self.tgtM = param
        
    def fit(self, df):
        self.ft_name = df.columns.values[0]
        if self.tgtM in ['mean', 'median', 'min', 'max']:
            tgt_v = eval('df[self.ft_name].%s()'%(self.tgtM))
        elif isinstance(self.tgtM, str):
            warnings.warn('str value input for missing fulfill, please check!')
            tgt_v = self.tgtM
            
        self.tgtV = tgt_v
            
        return self
            
    def transform(self, df):
        try:
            df = df.assign(**{self.ft_name:df[self.ft_name].fillna(self.tgtV)})
            return df
        except KeyError:
            warnings.warn('Attention: %s No matched features' %self.ft_name)
            return df[[self.ft_name]].fillna(self.tgtV)
    
class capMethods(object):
    def __init__(self, param):
        self.up = param.get('up')
        self.floor = param.get('floor')
        self.max_pct = param.get('max_pct')
        self.min_pct = param.get('min_pct')
        
    def _subcapMethods(self, x, s, b):
        if s is None and b is None:
            return x
        elif s is None and b is not None:
            return min(x, b)
        elif s is not None and b is None:
            return max(x, s)
        else:
            return min(b, max(x, s))
        
    def fit(self, df):
        self.ft_name = df.columns.values[0]
        if (self.up is None) and (self.max_pct is not None):
            up = df[self.ft_name].quantile(self.max_pct)
        else:
            up = self.up
            
        if (self.floor is None) and (self.min_pct is not None):
            floor = df[self.ft_name].quantile(self.min_pct)
        else:
            floor = self.floor
            
        self.upV = up
        self.floorV = floor
            
        return self
    
    def transform(self, df):
        if (self.up is None) and (self.floor is None):
            warnings.warn('Function dose not actually run')
        try:
            df = df.assign(**{self.ft_name:df[self.ft_name].apply(self._subcapMethods, s = self.floorV, b = self.upV)})
        except KeyError:
            warnings.warn('Attention: %s No matched features' %self.ft_name)
            ft_name = df.columns.values[0]
            df = df.assign(**{ft_name:df[ft_name].apply(self._subcapMethods, s = self.floorV, b = self.upV)})
        return df
    
class onehotMethods(object):
    def __init__(self, param):
        self.value_range = param.get('value_range')
        
    def fit(self, df):
        self.ft_name = df.columns.values[0]
        value_range = list(set(list(df[self.ft_name])))
        if self.value_range is None:
            self.value_rangeV = value_range
        else:
            self.value_rangeV = list(set(self.value_range).intersection(value_range))
            
        return self
    
    def transform(self, df):
        try:
            ft_name = self.ft_name
            df = df.assign(**{ft_name:df[ft_name].apply(lambda x: x if x in self.value_rangeV else None)})
        except KeyError:
            warnings.warn('Attention: %s No matched features' %self.ft_name)
            ft_name = df.columns.values[0]
            df = df.assign(**{ft_name:df[ft_name].apply(lambda x: x if x in self.value_rangeV else None)})
             
        if df[ft_name].isna().max() == 1:
            return pd.get_dummies(df[ft_name], prefix = ft_name, dummy_na = False).astype(object).astype(int)
        else:
            return pd.get_dummies(df[ft_name], prefix = ft_name, drop_first = True).astype(object).astype(int)
        
class setStrMethods(object):
    def __init__(self, param):
        self.strs2orders = param
        
    def fit(self, df):
        self.ft_name = df.columns.values[0]
        return self
    
    def transform(self, df):
        try:
            df = df.assign(**{self.ft_name:df[self.ft_name].apply(lambda x: self.strs2orders[x] if x in self.strs2orders.keys() else None)})
        except:
            warnings.warn('Attention: %s No matched features' %self.ft_name)
            ft_name = df.columns.values[0]
            df = df.assign(**{ft_name:df[ft_name].apply(lambda x: self.strs2orders[x] if x in self.strs2orders.keys() else None)})
            
        return df
    
class woeMethods(WoeFuncs):
    def __init__(self, param):
        super(woeMethods, self).__init__(param.get('pct_size'), param.get('max_grps'), param.get('chiq_pv'), \
                                         param.get('ifmono'), param.get('ifnan'), param.get('methods'))
        self.bins = param.get('bins')
        self.woes = param.get('woes')            
        self.type_info = param.get('type_info')
        
    def fit(self, df):
        #self.ft_name = df.columns.values[0]
        self.setTgt(df)
        if self.bins is None or self.woes is None:
            if self.type_info == 'str':
                self.strWoe_cal()
            elif self.type_info in ['int', 'float']:
                self.woe_cal()
            elif isinstance(self.type_info, dict):
                self._setStrValue(self.type_info, ifraise = False)
                self.woeDetail[self.ft_name]['str2orders'] = self.type_info
                self.woe_cal()
        else:
            self.woeDetail[self.ft_name]['bins'] = self.bins
            self.woeDetail[self.ft_name]['woes'] = self.woes
                
        self.freeMmy()
            
        return self
    
    def transform(self, df):
        self.setTgt(df)
        if isinstance(self.woeDetail[self.ft_name].get('str2orders'), dict):
            self._setStrValue(self.woeDetail[self.ft_name]['str2orders'], ifraise = False)
            self.woe_apply()
            
        elif isinstance(self.woeDetail[self.ft_name]['bins'], dict):
            self.strWoe_apply()
            
        else:
            self.woe_apply()
            
        mdf = self.getWoeCode()
        self.freeMmy()
        return mdf
    

class AllFtrProcess(object):
    """
    a typical methods indicator example:
    {
        "undo": [],
        "fill": {
            "MD002K": "median",
            "F0009": "median",
            "D012": 0,
            "insurance_score": "median",
            "probability": "median",
            "score": "median"
        },
        "cap": {
            "MD002K": {
                "up": 1
            }
        },
        "var2char": {
            "ft_test1": {
                "h": 3,
                "m": 2,
                "l": 1
            }
        },
        "onehot": {
            "ft_test3": {
                "value_range": [
                    "f",
                    "k"
                ]
            }
        },
        "woeCal": {
            "ft_test2": {
                "type_info": "str"
            },
            "ft_test4": {
                "type_info": {
                    "h": 3,
                    "l": 1,
                    "m": 2
                }
            },
            "MD04EY": {
                "bins": [
                    0,
                    1,
                    10
                ],
                "woes": {
                    "[0, 1)": 0.02077037730812839,
                    "[1, 10)": -0.2382850684756809
                }
            },
            "MD0038": {
                "type_info": "int"
            }
        }
    }
    """
    def __init__(self, path, pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
        self.params = {}
        self.params['pct_size'] = pct_size
        self.params['max_grps'] = max_grps
        self.params['chiq_pv'] = chiq_pv
        self.params['ifmono'] = ifmono
        self.params['ifnan'] = ifnan
        self.params['methods'] = methods
    
        smy = tools.getJson(path)
        self.undo_list = smy['undo']
        self.fill_list = smy['fill']
        self.cap_list = smy['cap'] #
        self.var2char_list = smy['var2char'] #should be a dictionary
        self.onehot_list = smy['onehot']
        self.woe_list = smy['woeCal']
        
    def fit(self, data):
        """
        此方法每一个特征只能进行一次处理；
        若想顺序调整则需要重写fit
        """
        all_methods = {}
        for k,v in self.fill_list.items():
            all_methods[k] = fillMethods(v).fit(data[[k]])
            
        for k,v in self.cap_list.items():
            all_methods[k] = capMethods(v).fit(data[[k]])
            
        for k,v in self.var2char_list.items():
            all_methods[k] = setStrMethods(v).fit(data[[k]])
            
        for k,v in self.onehot_list.items():
            all_methods[k] = onehotMethods(v).fit(data[[k]])
            
        for k,v in self.woe_list.items():
            woe_param = {ak:av for ak,av in self.params.items()}
            woe_param['type_info'] = v.get('type_info')
            woe_param['bins'] = v.get('bins')
            woe_param['woes'] = v.get('woes')
            all_methods[k] = woeMethods(woe_param).fit(data[[k, 'label']])
            
        self.all_methods = all_methods
        return self
            
    def transform(self, data, iflabel = True):
        all_ftrs = self.undo_list + list(self.all_methods.keys())
        if iflabel:
            data = data[all_ftrs+['label']]
        else:
            data = data[all_ftrs]
        try:
            with tqdm(self.all_methods.items()) as t:
                for f,m in t:
                    #print(f)
                    if f in self.onehot_list.keys():
                        data = pd.merge(left = data, right = m.transform(data[[f]]), right_index = True, left_index = True, how = 'left')
                    else:
                        data = data.assign(**{f:m.transform(data[[f]])})
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
                    
        #for f,m in self.all_methods.items():
        #    data = data.assign(**{f:m.transform(data[[f]])})
            
        data = data.drop(list(self.onehot_list.keys()), axis = 1)
            
        return data
    
    
    

#class FeatureProcessFuncs(AllWoeFuncs):
#    """docstring fs usprvFeatureProcess.
#    a specific dicionary, indicating processing methods for average single features is required here:
#    1.No Processing;
#    2.fillna with (mean, average, min, max, or any other value)；
#    3.one hot;
#    4.set quantitative values for str values;
#    5.woe coding for (str or numbers)
#    =======================================================
#    setData heritated from AllWoeFuncs will be used for all data s
#    """
#    def __init__(self, path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
#        super(FeatureProcessFuncs, self).__init__(path, pct_size, max_grps, chiq_pv, ifmono, ifnan, methods)
#        self.undo_list_fit = []
#        self.fill_list_fit = {}
#        self.cap_list_fit = {}
#        self.var2char_list_fit = {}
#        self.onehot_list_fit = {}
#        self.woe_list_fit = {}
#
#    def setTgt(self, df):
#        self.df = df
#        self.ft_name = list(df.columns.values)[0]
#
#    def fillmethods(self, value, inplace = False):
#        if isinstance(value, int) or isinstance(value, float):
#            tgt_v = value
#        elif value in ['mean', 'median', 'min', 'max']:
#            tgt_v = eval('self.df[self.ft_name].%s()'%value)
#        else:
#            tgt_v = value
#            warnings.warn('str value input for missing fulfill, please check!')
#        
#        self.fill_list_fit[self.ft_name] = tgt_v
#        if inplace:
#            self.df = self.df.fillna(tgt_v)
#        else:
#            self.mdf = self.df.fillna(tgt_v)
#
#    def _subcapMethods(self, x, s, b):
#        if s is None and b is None:
#            return x
#        elif s is None and b is not None:
#            return min(x, b)
#        elif s is not None and b is None:
#            return max(x, s)
#        else:
#            return min(b, max(x, s))
#
#    def capMethods(self, up = None, floor = None, max_pct = None, min_pct = None, inplace = False):
#        if (up is None) and (max_pct is not None):
#            up = self.df[self.ft_name].quantile(max_pct)
#
#        if (floor is None) and (min_pct is not None):
#            floor = self.df[self.ft_name].quantile(min_pct)
#
#        data = self.df.copy()
#
#        if (up is None) and (floor is None):
#            warnings.warn('Function dose not actually run')
#        data[self.ft_name] = data[self.ft_name].apply(self._subcapMethods, s = floor, b = up)
#        self.cap_list_fit[self.ft_name] = {'up':up, 'floor':floor}
#
#        if inplace:
#            self.df = data
#        else:
#            self.mdf = data
#
#    def onehot_sf(self, data = None, value_range = None):
#        ft_name = self.ft_name, data = self.df.copy()
#
#        if value_range is not None:
#            data[ft_name] = data['ft_name'].apply(lambda x: x if x in value_range else 'nan')
#
#        if 'nan' in data[ft_name]:
#            mdf = pd.get_dummies(data[ft_name])
#            mdf.columns = [ft_name+'_'+str(a) for a in mdf.columns.values]
#            mdf = mdf.drop(ft_name+'_nan')
#        else:
#            mdf = pd.get_dummies(data[ft_name], drop_first = True)
#            mdf.columns = [ft_name+'_'+str(a) for a in mdf.columns.values]
#            
#        self.onehot_list_fit[self.ft_name] = [a for a in list(set(list(data[ft_name]))) if (a != 'nan' and a is not None)]
#
#        self.mdf = mdf
#
#    def setStrMethods(self, strs2orders, inplace = False):
#        """
#        给予字符串变量可比较性；
#        strs2orders : 一个字典包含对应的字符串与数值对应关系
#        """
#        data = self.df.copy()
#        data[self.ft_name] = data[self.ft_name].apply(lambda x: strs2orders[x] if x in strs2orders.keys() else None)
#        if inplace:
#            self.df = data
#        else:
#            self.mdf = data
#            
#        self.var2char_list_fit[self.ft_name] = strs2orders
#
#    def RobustScaler(self, data, apply_data = None, **args):
#        """
#        using the scaler method provided by preprocessing, params are followed:
#        with_centering : boolean, True by default
#            If True, center the data before scaling.
#            This will cause ``transform`` to raise an exception when attempted on
#            sparse matrices, because centering them entails building a dense
#            matrix which in common use cases is likely to be too large to fit in
#            memory.
#
#        with_scaling : boolean, True by default
#            If True, scale the data to interquartile range.
#
#        quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
#            Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
#            Quantile range used to calculate ``scale_``.
#
#        copy : boolean, optional, default is True
#            If False, try to avoid a copy and do inplace scaling instead.
#            This is not guaranteed to always work inplace; e.g. if the data is
#            not a NumPy array or scipy.sparse CSR matrix, a copy may still be
#            returned.
#        """
#        prs = preprocessing.RobustScaler(**args).fit(data)
#        if apply_data is None:
#            self.mdf = prs.transform(data)
#        else:
#            self.mdf = prs.transform(apply_data)
#
#    def minmax_scaler(self, data, apply_data = None, **args):
#        """
#        another funcstions that can be directorly inheritated from preprocessing
#        """
#        pass
#
#    def feature_egineer(self):
#        """
#        save for feature_egineer
#        """
#        pass
#
#    def getMdf(self):
#        return self.mdf
#
#class FeatureProcess(FeatureProcessFuncs):
#    """docstring for ."""
#    def __init__(self, path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
#        super(FeatureProcess, self).__init__(path, pct_size, max_grps, chiq_pv, ifmono, ifnan, methods)
#
#    def setData(self, data):
#        self.data = data
#
#    def collectSummary(self, path):
#        smy = tools.getJson(path)
#        self.undo_list = smy['undo']
#        self.fill_list = smy['fill']
#        self.cap_list = smy['cap'] #
#        self.var2char_list = smy['var2char'] #should be a dictionary
#        self.onehot_list = smy['onehot']
#        self.woe_list = smy['woeCal']
#
#    def allProcess(self, vrs, ifsave = True):
#        """
#        vrs:indicating the version of woe coding in this function
#        """
#        """
#        缺失值填充
#        """
#        for i in self.fill_list.keys():
#            self.setTgt(self.data[[i]])
#            self.fillmethods(self.fill_list[i], inplace = False)
#            #mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')
#            
#        mdf = mdf.assign(**{a:self.mdf[a] for a in list(self.mdf.columns.values)})
#
#        """
#        去除极值
#        """
#        for i in self.cap_list.keys():
#            self.setTgt(self.data[[i]])
#            up = self.cap_list[i].get('up'); floor = self.cap_list[i].get('floor')
#            max_pct = self.cap_list[i].get('max_pct'); min_pct = self.cap_list[i].get('min_pct')
#            self.capMethods(up, floor, max_pct, min_pct, inplace = False)
#            #mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')
#            
#        mdf = mdf.assign(**{a:self.mdf[a] for a in list(self.mdf.columns.values)})
#
#        """
#        处置字符串型变量
#        """
#        for i in self.var2char_list.keys():
#            self.setTgt(self.data[[i]])
#            self.setStrMethods(self.var2char_list[i], inplace = False)
#            #mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')
#            
#        mdf = mdf.assign(**{a:self.mdf[a] for a in list(self.mdf.columns.values)})
#
#        """
#        woe编码:precalculated iv info needed
#        """
#        self.setData(self.data[list(self.woe_list.keys())+['label']])
#        self.setFtrs(self.woe_list)
#        try:
#            self.AllWoeCollects(vrs)
#        except:
#            warnings.warn('no given woe info detetd, all data will be used to calculate woe first')
#            self.AllWoeCals(vrs)
#        self.AllWoeApl(ifsave = False)
#
#        #mdf = pd.merge(left = mdf, right = self.getMdfData().drop('label', axis = 1), left_index = True, right_index = True)
#        mdf = mdf.assign(**{a:self.getMdfData()[a] for a in list(self.woe_list.keys())})
#        
#        """
#        Onehot变量处理
#        """
#        for i in self.onehot_list.keys():
#            self.setTgt(self.data[[i]])
#            self.onehot_sf(self.onehot_list[i].get('value_range'))
#            #mdf = pd.merge(left = mdf, right = self.mdf, left_index = True, right_index = True, how = 'left')
#        
#        mdf = mdf.drop(list(self.onehot_list.keys()), axis = 1)
#        mdf = mdf.assign(**{a:self.mdf[a] for a in list(self.mdf.columns.values)})
#
#        if ifsave:
#            mdf.to_csv(self.path+'/FeatureModify/DataToModel.csv')
#        self.finalData = mdf
#
#    def getFinalData(self):
#        try:
#            return self.finalData
#        except:
#            return None
