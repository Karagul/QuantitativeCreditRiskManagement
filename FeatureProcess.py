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
    
class putNaMethods(object):
    def __init__(self,na_list=[],na_dict={}):
        self.na_list=na_list
        self.na_dict=na_dict
        if not (isinstance(na_list,list) and isinstance(na_dict,dict)):
            warnings.warn('Attention: na_dict and na_list type is wrong')
        elif len(na_list)==0 and len(list(na_dict.keys()))==0:
            warnings.warn('Attention: na_dict and na_list has no elements')
        else:
            pass
       
    def fit(self, df):
        self.ft_name = df.columns.values[0]
        return self
    
    def transform(self, df):
        if df.columns.values[0]!=self.ft_name:
            warnings.warn('Attention: fearure in transform and feature in fit not match')
        else:
            pass
        
        if len(list(self.na_dict.keys()))>0 and self.ft_name in list(self.na_dict.keys()):
            df = df.assign(**{self.ft_name:df[self.ft_name].apply(lambda x: np.nan if x in self.na_dict[self.ft_name].values() else x)})  
        elif len(self.na_list)>0:
            df = df.assign(**{self.ft_name:df[self.ft_name].apply(lambda x: np.nan if x in self.na_list else x)})
        else:
            warnings.warn('Attention: no treat in %s'%self.ft_name)
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
        else:
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
            if df[self.ft_name].isna().sum() >= 1:
                if 'nan' in self.strs2orders.keys():
                    df = df.fillna(self.strs2orders['nan'])
                else:
                    warnings.warn('Attention: setStrMethods Nan value happend!')
                
        except:
            warnings.warn('Attention: %s No matched features' %self.ft_name)
            ft_name = df.columns.values[0]
            df = df.assign(**{ft_name:df[ft_name].apply(lambda x: self.strs2orders[x] if x in self.strs2orders.keys() else self.strs2orders['nan'])})
            if df[ft_name].isna().sum() >= 1:
                if 'nan' in self.strs2orders.keys():
                    df = df.fillna(self.strs2orders['nan'])
                else:
                    warnings.warn('Attention: setStrMethods Nan value happend!')
            
        return df
    
class woeMethods(WoeFuncs):
    def __init__(self, param):
        super(woeMethods, self).__init__(param.get('pct_size'), param.get('max_grps'), param.get('chiq_pv'), \
                                         param.get('ifmono'), param.get('keepnan'), param.get('methods'))
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
            self.woe_apply(keepnan = self.keepnan)
            
        elif isinstance(self.woeDetail[self.ft_name]['bins'], dict):
            self.strWoe_apply(keepnan = self.keepnan)
            
        else:
            self.woe_apply(keepnan = self.keepnan)
            
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
    def __init__(self, path, pct_size = 0.03, max_grps = 5, chiq_pv = 0.05, ifmono = True, keepnan = True, methods = 'tree'):
        self.params = {}
        self.params['pct_size'] = pct_size
        self.params['max_grps'] = max_grps
        self.params['chiq_pv'] = chiq_pv
        self.params['ifmono'] = ifmono
        self.params['keepnan'] = keepnan
        self.params['methods'] = methods
        
        if isinstance(path, dict):
            smy = path
        else:
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
            if k in data.columns:
                all_methods[k] = fillMethods(v).fit(data[[k]])
            
        for k,v in self.cap_list.items():
            if k in data.columns:
                all_methods[k] = capMethods(v).fit(data[[k]])
            
        for k,v in self.var2char_list.items():
            if k in data.columns:
                all_methods[k] = setStrMethods(v).fit(data[[k]])
            
        for k,v in self.onehot_list.items():
            if k in data.columns:
                all_methods[k] = onehotMethods(v).fit(data[[k]])
            
        for k,v in self.woe_list.items():
            if k in data.columns:
                #print(k)
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
                    if f in data.columns:
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
    
