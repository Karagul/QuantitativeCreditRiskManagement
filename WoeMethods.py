import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from scipy import stats

import tqdm
from sklearn.tree import DecisionTreeClassifier

from tools import *


class bins_method_funcs(object):
    def __init__(self, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05):
        """
        单因子的WOE处理方法，包含三种分享方式，WOE编码的具体防范
        """
        self.argms = {}
        self.argms['pct_size'] = pct_size
        self.argms['max_grps'] = max_grps
        self.argms['chiq_pv'] = chiq_pv

    def setTgt(self, df):
        self.raw = df
        self.ft_name, _ = self.raw.columns.values

    def setParams(self, params):
        for i in params.keys():
            if i not in ['pct_size', 'max_grps', 'chiq_pv']:
                warnings.warn('Invalid Parameters, Please Check')
            self.argms[i] = params[i]

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
        1.当分组的单调性不满足的时候，优先合并chiq检验最不显著的分组；
        2.重复以上步骤
        3.cuts要保持单调
        """
        df = tgt.copy().dropna()
        ft_name, _ = df.columns.values

        df['grp'] = pd.cut(df[ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(df)

        #单调性检验，同时合并卡方最不显著的分组
        while len(set([chis[i] < chis[i+1] for i in range(len(chis))]))>1:
            lct = chis.index(max(chis))
            cuts.remove(cuts[lct+1])
            chis = self._chi_cal_func(df, cuts)

        return cuts

    def _setStrValue(self, strs2orders):
        """
        给予字符串变量可比较性
        """
        self.raw[self.ft_names] = self.raw[self.ft_names].apply(lambda x: strs2orders[x] if x in strs2orders.keys() else np.nan)

    def qlt_bins_func(self):
        """
        对于定性类的特征进行分箱统计
        对于无序的特征进行WOE编码有可能产生过拟合的可能，并不建议直接使用
        """
        tmp = self.raw.copy().dropna()
        values = list(set(list(tmp[self.ft_name].values)))

        strBins = {}
        for i in range(len(values)):
            strBins[values[i]] = i

        self.bins = {self.ft_name: strBins}

    def tree_bins_func(self):
        """
        基于决策树（信息熵）的分组
        1.max_grps控制最大分组的个数；
        2.pct_size控制每组最低的样本占比
        """
        tmp = self.raw.copy().dropna()
        smp_size = np.int(len(tmp)*self.argms['pct_size'])+1

        #当特征的最大取值占比超过阈值时，不做进一步区分，只分为2组
        #以决策树为分组的基准工具
        min_check = df.groupby(self.ft_name).count()['label'].max()
        if min_check >= len(tmp) - smp_size:
            clf = DecisionTreeClassifier(max_leaf_nodes = 2)
        else:
            clf = DecisionTreeClassifier(min_samples_leaf = smp_size, max_leaf_nodes = self.argms['max_grps'])
        clf.fit(tmp[[self.ft_name]], tmp['label'])

        tmp['grp_prd'] = clf.apply(tmp[[self.ft_name]])

        grp_info = tmp.groupby('grp_prd').min()
        grp_info.sort_values(self.ft_name, inplace = True, ascending = True)
        cuts = list(grp_info[self.ft_name]) + [df[self.ft_name].max()+1]

        self.bins = {self.ft_name:cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}

    def dist_bins_funcs(self):
        """
        基于特征取值范围进行分组，需要应对分布不均匀的情况
        目前看上去没有必要保留此函数
        """
        pass

    def freq_bins_funcs(self):
        """
        基于频率的分组方式：
        1.grps控制分组的个数；
        2.pct_size控制任意分组的最小总体样本占比
        """
        grps = self.argms['max_grps']
        pct_size = self.argms['pct_size']

        tmp = self.raw.copy().dropna()
        #在已知分组数的前提下，允许任意分组样本占总体样本向下浮动一定的比例
        pct_size = min(pct_size, 1.0/grps/1.5)
        smp_size = np.int(len(tmp)*pct_size)+1

        prm_cuts = [tmp[self.ft_name].quantile(1.0/grps * a, interpolation = 'lower') for a in range(grps)]
        prm_cuts += [tmp[self.ft_name].max()+1]

        prm_cuts = list(set(prm_cuts))
        prm_cuts.sort()

        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = prm_cuts, index = range(len(prm_cuts)-1), right = False)
        stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()

        rlts = []

        #最后检查分组后分组样本的占比情况，当样本个数欧过少时合并
        while stat[self.ft_name].min() < smp_size:
            tgt_loc = np.argmin(stat[self.ft_name])
            if tgt_loc == 0:
                prm_cuts.remove(prm_cuts[tgt_loc+1])
            elif tgt_loc == len(prm_cuts)-2:
                prm_cuts.remove(prm_cuts[tgt_loc-1])
            elif stat.loc[tgt_loc-1,self.ft_name] < stat.loc[tgt_loc+1, self.ft_name]:
                prm_cuts.remove(prm_cuts[tgt_loc])
            else:
                prm_cuts.remove(prm_cuts[tgt_loc+1])

            tmp['grp'] = pd.cut(tmp[self.ft_name], bins = prm_cuts, index = range(len(prm_cuts)-1), right = False)
            stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()

        self.bins = {self.ft_name:prm_cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}


    def chiq_bins_func(self):
        """
        通过chiq进行的分箱，先用频率分箱的方式分出较多组，后续迎chiq的方式进行合并
        1.grps控制初始频率分箱的组数；
        2.cuts可以指定初始分箱方式；
        3.pct_size控制最小分组样本占整体的最小比例；
        4.pv控制是否合并分箱的阈值
        """
        grps = self.argms['max_grps']
        pct_size = self.argms['pct_size']
        pv = self.argms['chiq_pv']

        tmp = self.raw.copy().dropna()

        #先使用均匀分组当方式分割为20组
        self.freq_bins_funcs(grps = 20, pct_size = 0.03)

        cuts = self.freq_bins[self.ft_name]
        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(tmp)
        #以卡方为基准进行分箱合并
        while max(chis) > pv:
            tgt = chis.index(max(chis))
            cuts.remove(cuts[tgt+1])
            if len(cuts)<=2:
                break
            else:
                tmp['grp'] = pd.cut(tmp[self.ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
                chis = self._chi_cal_func(tmp)

        self.bins = {self.ft_name:cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}

    def mono_bins_func(self):
        """
        检验任意分组的单调性：
        1.用卡方的方式确保单调性
        """
        bins = self.bins

        tmp = data.dropna()

        tmp[self.ft_name] = tmp[self.ft_name].apply(self._llt_cap_func, (cap_info['min'], cap_info['max']))
        cuts = self._bins_merge_chiq(tmp, bins)
        self.bins = {self.ft_name:cuts}

    def getWoeBins(self):
        return self.bins

class WoeFuncs(bins_method_funcs):
    """docstring for WoeFuncs."""
    def __init__(self, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
        super(WoeFuncs, self).__init__(pct_size, max_grps, chiq_pv)
        self.ifmono = True; self.ifnan = True
        self.methods = methods

        self.all_woe_info = {}
        self.allBins = {}

    def woe_cal(self, data = None):
        """
        计算特征的IV值及相应分箱的WOE编码
        1.data：意味着可以传输外部数据进行计算；
        2.ifiv: 控制是否返回最终的IV值；
        3.ifnan: 控制是否进行空值处理；
        4.methods: 控制分箱方法，只有‘tree’， ‘chiq’， ‘freq’三种可选；
        """
        methods = self.methods
        ifmono = self.ifmono
        ifnan = self.ifnan

        if data is None:
            data = self.raw.copy()

        try:
            bins = self.bins[self.ft_name]
        except:
            if methods == 'tree':
                self.tree_bins_func()
            elif methods == 'chiq':
                self.chiq_bins_func()
            elif methods == 'freq':
                self.freq_bins_func()
            else:
                raise ValueError('Invalid Input Methods')

        if self.argms['ifmono']:
            self.mono_bins_func()

        bins = self.bins[self.ft_name]
        cap_info = self.cap_info

        tmp = data.dropna()
        #确保test及其他不会出现train上没有见到过当奇异值
        tmp[self.ft_name] = tmp[self.ft_name].apply(self._llt_cap_func, (cap_info['min'], cap_info['max']))
        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = bins, right = False).apply(str)
        stat = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        #是否保留空值当统计
        if ifnan:
            rlts = list(stat.values) + [['nan', data[data[self.ft_name].isna()]['label'].sum(), len(data[data[self.ft_name].isna()])]]
        else:
            rlts = stat
        woe = pd.DataFrame(rlts, columns = [self.ft_name, 'bad', 'size'])

        #IV及woe值的计算
        woe['good'] = woe['size'] - woe['bad']
        woe['woe'] = ((woe['bad']/bad)/(woe['good']/good)).apply(np.int)
        woe['iv'] = (woe['bad']/bad - woe['good']/good) * woe['woe']
        woe['bad_pct'] = woe['bad']/woe['size']

        tmp_dict = {}
        for i in woe[self.ft_name].values:
            tmp_dict[i] = woe[woe[self.ft_name]==i].loc[0]['woe']
        self.all_woe_info[self.ft_name] = tmp_dict
        self.allBins[self.ft_name] = self.getBins()

        self.iv_stat = {}
        self.iv_stat['woe_info'] = woe
        self.iv_stat['iv_value'] = woe['iv'].sum()

    def strWoe_cal(self, data = None, cuts = None, ifnan = True):
        """
        对于定性类的特征进行woe计算
        """
        if cuts is None:
            try:
                cuts = self.bins[self.ft_name]
            except:
                self.qlt_bins_func()

        cuts = self.bins[self.ft_name]

        if data is None:
            tmp = self.raw.copy()
        else:
            tmp = data.copy()

        #合并特定类别限制分组数
        cuts = self._strWoe_merge(tmp, self.ft_anme, cuts, self.argms['max_grps'])

        tmp['grp'] = tmp[self.ft_name].apply(lambda x: cuts[x] if x in cuts.keys() else 'nan')
        if not ifnan:
            tmp = tmp[tmp[self.ft_name]!='nan']

        woe = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        woe.columns = [self.ft_name, 'bad', 'size']

        woe['good'] = woe['size'] - woe['bad']
        woe['woe'] = ((woe['bad']/bad)/(woe['good']/good)).apply(np.int)
        woe['iv'] = (woe['bad']/bad - woe['good']/good) * woe['woe']
        woe['bad_pct'] = woe['bad']/woe['size']

        tmp_dict = {}
        for i in woe[self.ft_name].values:
            tmp_dict[i] = woe[woe[self.ft_name]==i].loc[0]['woe']
        self.all_woe_info[self.ft_name] = tmp_dict
        self.allBins[self.ft_name] = self.getBins()

        self.iv_stat = {}
        self.iv_stat['woe_info'] = woe
        self.iv_stat['iv_value'] = woe['iv'].sum()

    def _strWoe_merge(df, ft_name, org_bins, max_grps):
        df['grp'] = df[ft_name].apply(lambda x: org_bins[x] if x in org_bins.keys() else 'nan')
        df = df[df['grp']!='nan']

        woe = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        woe.columns = [ft_name, 'bad', 'size']
        woe['bad_pct'] = woe['bad']/woe['size']

        max_grp = woe_info[ft_name].max()
        while len(woe_info)>max_grps:
            loc = woe_info['bad_pct'].diff(periods = 1).loc[1:].apply(abs).idxmin()
            m1, m2 = woe.loc[loc-1, ft_name], woe.loc[loc, ft_name]

            tgt = [akey for akey, avalue in org_bins if avalue == m1][0]
            org_bins[tgt] = m2
            df['grp'] = df[ft_name].apply(lambda x: org_bins[x] if x in org_bins.keys() else 'nan')
            woe = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
            woe.columns = [ft_name, 'bad', 'size']
            woe['bad_pct'] = woe['bad']/woe['size']

    def woe_apply(self, data = None, ifnan = False):
        """
        用WOE编码替换分组信息
        """
        if data is None:
            data = self.raw.copy()

        if ifnan:
            data = data.dropna()

        try:
            cuts = self.allBins[self.ft_name]
            woe_info = self.all_woe_info[self.ft_name]
        except:
            self.woe_cal()
            cuts = self.allBins[self.ft_name]
            woe_info = self.all_woe_info[self.ft_name]

        data['grp'] = pd.cuts(data[self.ft_name], bins = cuts, right = False).apply(str)
        data['grp'] = data['grp'].fillna('nan')
        data[self.ft_name] = data['grp'].apply(lambda x: woe_info[x])

        self.mdf = data

    def strWoe_apply(self, data = None, ifnan = False):
        """
        对于定性类特征进行WOE编码替换
        """
        if data is None:
            data = self.raw.copy()

        if ifnan:
            data = data.dropna()

        try:
            cuts = self.allBins[self.ft_name]
            woe_info = self.all_woe_info[self.ft_name]
        except:
            self.woe_cal_q()
            cuts = self.allBins[self.ft_name]
            woe_info = self.all_woe_info[self.ft_name]

        data['grp'] = data[self.ft_name].apply(lambda x: cuts[x] if x in cuts.keys() else 'nan')
        data[self.ft_name] = data['grp'].apply(lambda x: woe_info[x])

        self.mdf = data

    def getWoeCode(self):
        return self.mdf[[self.ft_name]]

    def getWoeInfo(self):
        return self.iv_stat['woe_info']

    def getIVinfo(self):
        return self.iv_stat['iv_value']

class AllWoeFuncs(WoeFuncs):
    """docstring for AllWoeFUncs:
       1.check setFtrs and setData functions before any calculation
    """
    def __init__(self, path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
        super(AllWoeFuncs, self).__init__(pct_size, max_grps, chiq_pv, ifmono, ifnan, methods)
        self.path = path

    def setFtrs(self,ftrs):
        """
        ftrs needs to be a dictionary indicating the processing methods used by WoeFuncs
        'str' for strWoe_cal
        'int' for Woe_cal
        a specific dictionary is needed if the str values actually have a meaningful order
        """
        if len(set(ftrs.keys()) - set(self.data.columns.values))>0:
            raise ValueError('Features Not Match !!')
        else:
            self.ftrs = ftrs

    def setData(self, data):
        self.data = data

    def AllWoeCals(self, vrs):
        """
        vrs for version control
        """
        ivs = {}
        for i in self.ftrs.keys():
            self.setTgt(self.data[[i, 'label']])
            if self.ftrs.keys[i] == 'str':
                self.strWoe_cal()
            elif self.ftrs.keys[i] == 'int':
                self.woe_cal()
            elif isinstance(self.ftrs.keys[i], dict):
                self._setStrValue(self.ftrs.keys[i])
                self.woe_cal()

            ivs[i] = self.getIVinfo()

        putFile(self.path+'IVstat/IVs_'+vrs+'.json', ivs)
        putFIle(self.path+'IVstat/binsInfo_'+vrs+'.json', self.allBins)
        putFile(self.path+'IVstat/IVDetails_'+vrs+'.json', self.all_woe_info)

    def AllWoeCollects(self, vrs):
        """
        collect calculated woe infos
        """
        self.allBins = getFiles(self.path + 'IVstat/binsInfo_'+vrs+'.json')
        self.all_woe_info = getFile(self.path + 'IVDetails_'+vrs+'.json')

    def AllWoeApl(self, ifsave = True):
        try:
            if len(set(self.ftrs.keys())-set(self.all_woe_info.keys())) > 0:
                raise ValueError('Run func AllWoeCals first')
            else:
                pass
        except:
            raise ValueError('Run func AllWoeCals first')

        if 'label' in self.data.columns.values:
            self.data_woe = self.data[['label']]
        else:
            self.data['label'] = 1
            self.data_woe = self.data[['label']]

        for i in self.ftrs.keys():
            self.setTgt(self.data[[i, 'label']])
            if self.ftrs.keys[i] == 'str':
                self.strWoe_apply()
            elif self.ftrs.keys[i] == 'int':
                self.Woe_apply()
            elif isinstance(self.ftrs.keys[i], dict):
                self._setStrValue(self.ftrs.keys[i])
                self.Woe_apply()

            self.woe_apply()
            self.data_woe = pd.merge(left = self.data_woe, right = self.getWoeCode(), left_index = True, right_index = True, how = 'left')

        if 'label' not in self.data.columns.values:
            self.data_woe.drop('label', axis = 1, inplace = True)

        self.data_woe = data_woe
        if ifsave:
            self.data_woe.to_csv(path+'IVstat/ftrs_woe_code.csv')

    def getMdfData(self):
        return self.data_woe
