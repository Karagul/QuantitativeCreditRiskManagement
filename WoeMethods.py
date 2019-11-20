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
        """
        单因子的WOE处理方法，包含三种分享方式，WOE编码的具体防范
        """
        self.raw = df
        self.tree_bins = None
        self.freq_bins = None
        self.chiq_bins = None
        self.mono_bins = None
        self.all_woe_info = {}
        self.all_woe_mono_info = {}

    def _chi_cal_func(data):
        """
        所有相邻分组直接的卡方计算
        """
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
        """
        极值的处理函数
        """
        return max(s, min(x, b))

    def _bins_merge_chiq(tgt, cuts):
        """
        基于卡方的单调性保证：
        1.当分组的单调性不满足的时候，优先合并chiq检验最不显著的分醉；
        2.重复以上步骤
        """
        df = tgt.copy().dropna()
        ft_name, _ = df.columns.values

        df['grp'] = pd.cut(df[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(df)

        while len(set([chis[i] < chis[i+1] for i in range(len(chis))]))>1:
            lct = chis.index(max(chis))
            cuts.remove(cuts[lct+1])
            chis = self._chi_cal_func(df)

        return cuts

    def tree_bins_func(max_grps = 5, pct_size = 0.05):
        """
        基于决策树（信息熵）的分组
        1.max_grps控制最大分组的个数；
        2.pct_size控制每组最低的样本占比
        """
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
        self.cap_info = {'max': tmp[ft_name].max(), 'min':tmp[ft_name].min()}

    def freq_bins_funcs(grps = 10, pct_size = 0.05):
        """
        基于频率的分组方式：
        1.grps控制分组的个数；
        2.pct_size控制最小分组的最小可行比例
        """
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

    def chiq_bins_func(grps = 20, cuts = None, pct_size = 0.03, pv = 0.05):
        """
        通过chiq进行的分箱，先用频率分箱的方式分出较多组，后续迎chiq的方式进行合并
        1.grps控制初始频率分箱的组数；
        2.cuts可以指定初始分箱方式；
        3.pct_size控制最小分组样本占整体的最小比例；
        4.pv控制是否合并分箱的阈值
        """
        tmp = self.raw.copy().dropna()
        ft_name, _ = tmp.columns.values

        self.freq_bins_funcs(grps = grps, pct_size = pct_size)

        cuts = self.freq_bins[ft_name]
        tmp['grp'] = pd.cut(tmp[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(tmp)
        while max(chis) > pv:
            tgt = chis.index(max(chis))
            cuts.remove(cuts[tgt+1])
            if len(cuts)<=2:
                break
            else:
                tmp['grp'] = pd.cut(tmp[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
                chis = self._chi_cal_func(tmp)

        self.chiq_bins = {ft_name:cuts}

    def woe_cal(data = None, ifiv = False, ifnan = True, methods = 'tree', code = True):
        """
        计算特征的IV值及相应分箱的WOE编码
        1.data：意味着可以传输外部数据进行计算；
        2.ifiv: 控制是否返回最终的IV值；
        3.ifnan: 控制是否进行空值处理；
        4.methods: 控制分箱方法，只有‘tree’， ‘chiq’， ‘freq’三种可选；
        5.code：保留
        """
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

        tmp_dict = {}
        for i in woe[ft_name].values:
            tmp_dict[i] = woe[woe[ft_name]==i].loc[0]['woe']
        self.all_woe_info[ft_name] = tmp_dict

        if ifiv == True:
            return woe['iv'].sum()
        else:
            return woe

    def woe_mono_cal(data = None, ifiv = False, ifnan = True, methods = 'tree', code = True):
        """
        计算特征的IV值及相应分箱的WOE编码，保证分箱单调性
        1.data：意味着可以传输外部数据进行计算；
        2.ifiv: 控制是否返回最终的IV值；
        3.ifnan: 控制是否进行空值处理；
        4.methods: 控制分箱方法，只有‘tree’， ‘chiq’， ‘freq’三种可选；
        5.code：保留
        """
        if data is None:
            data = self.raw.copy()

        if methods == 'tree':
            bins = self.tree_bins_func
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
        cuts = self._bins_merge_chiq(tmp, bins)
        self.mono_bins = {ft_name:cuts}
        tmp['grp'] = pd.cut(tmp[ft_name], bins = cuts, right = False)
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

        tmp_dict = {}
        for i in woe[ft_name].values:
            tmp_dict[i] = woe[woe[ft_name]==i].loc[0]['woe']
        self.all_woe_mono_info[ft_name] = tmp_dict

        if ifiv == True:
            return woe['iv'].sum()
        else:
            return woe

    def woe_apply(data = None, cuts = None, woe_info = None, mothed = 'mono', ifna = True):
        """
        用WOE编码替换分组信息
        """
        if data is None:
            data = self.raw.copy()

        if ifna:
            data = data.dropna()

        ft_name, _ = data.columns.values
        if method == 'mono':
            cuts = self.mono_bins
            woe_info = self.all_woe_mono_info[ft_name]
        elif method == 'tree':
            cuts = self.tree_bins
            woe_info = self.all_woe_info[ft_name]
        elif method == 'chiq':
            cuts = self.chiq_bins
            woe_info = self.all_woe_info[ft_name]
        elif method == 'freq':
            cuts = self.freq_bins
            woe_info = self.all_woe_info[ft_name]

        data['grp'] = pd.cuts(data[ft_name], bins = cuts, right = False)
        data['grp'] = data['grp'].fillna('nan')
        data['woe_code'] = data['grp'].apply(lambda x: woe_info[x])

        self.data = data

    def func_pos_save():
        pass
