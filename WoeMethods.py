import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from scipy import stats

from dateutil.parser import parse
from pandas.tseries.offsets import MonthBegin, Day, MonthEnd
import statsmodels.api as sm
import tqdm

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import preprocesssing

class woe_methods_funcs(object):
    def __init__(df):
        self.raw = df
        self.tree_bins = None
        self.freq_bins = None
        self.chiq_bins = None
        pass

    def _chi_cal_func(data):
        names = list(data.columns.values)
        names.remove('grp')
        names.remove('label')
        tgt = names[0]
        stat = data.groupby(['grp', 'label'], as_index = False).count()

        grps = list(tmp['grp'].values)
        grps = list(set(grps))
        grps.sort()

        chis = []

        for i in range(len(grps)-1):
            tmp = stat[(stat['grp']==grps[i])|(stat['grp']==grps[i+1])]
            piv = tmp.pivot(index = 'grp', columns = 'label', values = tgt)
            chis += stats.chisquare(piv.loc[grps[i]], piv.loc[grps[i+1]])[0]

        return chis

    def _llt_cap_func(x, s, b):
        return max(s, min(x, b))

    def tree_bins_func(max_grps = 5, pct_size = 0.05):
        tmp = self.raw.copy().dropna()
        smp_size = np.int(len(tmp)*pct_size)+1

        ft_name, _ = tmp.columns.values

        min_check = df.groupby(ft_name).count()['label'].max()

        clf = DecisionTreeClassifier(min_samples_leaf = smp_size, max_leaf_nodes = max_grps)
        if min_check >= len(tmp) - smp_size:
            clf = DecisionTreeClassifier(max_leaf_nodes = 2)
        clf.fit(tmp[[ft_name]], tmp['label'])

        tmp['grp_prd'] = clf.apply(tmp[[ft_name]])

        grp_info = tmp.groupby('grp_prd').min()
        grp_info.sort_values(ft_name, inplace = True, ascending = True)
        cuts = list(grp_info[ft_name]) + [df[ft_name].max()+1]

        self.tree_bins = {ft_name:cuts}
        self.cap_info = {'max':}

    def freq_bins_funcs(grps = 10, pct_size = 0.05):
        tmp = self.raw.copy().dropna()
        pct_size = min(pct_size, 1.0/grps/1.5)

        ft_name, _ = tmp.columns.values

        prm_cuts = [tmp['ft_name'].quantile(1.0/grps * a, interpolation = 'lower') for a in range(grps)]
        prm_cuts += [tmp['ft_name'].max()+1]

        prm_cuts = list(set(prm_cuts))
        prm_cuts.sort()

        tmp['grp'] = pd.cut(tmp[ft_name], bins = prm_cuts, index = range(len(prm_cuts)-1), right = False)
        stat = tmp[['grp', ft_name]].groupby('grp', as_index = True).count()

        rlts = []
        for i in range(len(prm_cuts)-1):
            if stat.loc[i] <= pct_size:
                if i == 0:
                    rlts += [prm_cuts[i+1]]
                elif i == len(prm_cuts)-2:
                    rlts += [prm_cuts[i-1]]
                elif stat.loc[i-1] < stat.loc[i+1]:
                    rlts += [prm_cuts[i-1]]
                else:
                    rlts += [prm_cuts[i+1]]
            else:
                rlts += [prm_cuts[i]]
        rlts = list(set(rlts))
        rlt.sort()

        self.freq_bins = {ft_name:rlts+[max(prm_cuts)]}

    def chiq_bins_func(grps = 20, pct_size = 0.03, pv = 0.05):
        tmp = self.raw.copy().dropna()
        ft_name, _ = tmp.columns.values

        self.freq_bins_funcs(grps = grps, pct_size = pct_size)

        cuts = self.freq_bins[ft_name]
        tmp['grp'] = pd.cut(tmp[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(tmp)
        while max(chis) > pv:
            tgt = chis.index(max(chis))
            cuts.remove(cuts[tgt])
            if len(cuts)<=2:
                break
            else:
                tmp['grp'] = pd.cut(tmp[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
                chis = self._chi_cal_func(tmp)

        self.chiq_bins = {ft_name:cuts}

    def woe_apply(data = None, ifiv = False, ifnan = True, methods = 'tree', code = True):
        if data is None:
            data = self.raw.copy()

        if methods == 'tree':
            bins = self.tree_bins
            cap_info = self.cap_info
        elif methods == 'chiq':
            bins = self.chiq_bins
            cap_info = self.cap_info
        elif methods == 'freq':
            bins = self.freq_bins
            cap_info = self.cap_info
        else:
            raise ValueError('Invalid Input Methods')

        tmp = data.dropna()
        ft_name, _ = tmp.columns.values
        tmp[ft_name] = tmp[ft_name].apply(self._llt_cap_func, (cap_info['min'], cap_info['max']))
        tmp['grp'] = pd.cut(tmp[ft_name], bins = bins, right = False)
        stat = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        if ifnan:
            rlts = list(stat.values) + [['nan', data[data[ft_name].isna()]['label'].sum(), len(data[data[ft_name].isna()])]]
        else:
            rlts = stat
        woe = pd.DataFrame(rlts, columns = [ft_name, 'bad', 'size'])

        woe['good'] = woe['size'] - woe['bad']
        woe['woe'] = ((woe['bad']/bad)/(woe['good']/good)).apply(np.int)
        woe['iv'] = (woe['bad']/bad - woe['good']/good) * woe['woe']
        woe['bad_pct'] = woe['bad']/woe['size']

        if ifiv == True:
            return woe['iv'].sum()
        else:
            return woe
