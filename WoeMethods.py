import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from scipy import stats
import warnings

from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier


import tools


class bins_method_funcs(object):
    """
    提供分箱方法的对象，最终反馈的是分组对应的分箱以及特征取值区间
    """
    def __init__(self, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05):
        """
        单因子的WOE处理方法，包含三种分享方式，WOE编码的具体防范
        pct_size：控制分组的最小占比
        max_grps:控制最大分组数
        chiq_pv:卡方分组的合并基准
        """
        self.argms = {}
        self.argms['pct_size'] = pct_size
        self.argms['max_grps'] = max_grps
        self.argms['chiq_pv'] = chiq_pv

    def setTgt(self, df):
        """
        df:必须是一个两列的DataFrame，其中第一列是特征，第二列是label
        """
        self.raw = df
        self.ft_name = list(self.raw.columns.values)[0]

    def setParams(self, params):
        """
        与self.argms对应，可以重新设置参数
        """
        for i in params.keys():
            if i not in ['pct_size', 'max_grps', 'chiq_pv']:
                warnings.warn('Invalid Parameters, Please Check')
            self.argms[i] = params[i]

    def _chi_cal_func(self, data):
        """
        所有相邻分组直接的卡方计算
        包含ftr, label, 以及已经用过pd.cut之后的‘grp’特征
        """
        names = list(data.columns.values)
        names.remove('grp')
        names.remove('label')
        tgt = names[0]
        stat = data.groupby(['grp', 'label'], as_index = False).count()

        grps = list(stat['grp'].values)
        grps = list(set(grps))
        grps.sort()

        chis = []

        for i in range(len(grps)-1):
            tmp = stat[(stat['grp']==grps[i])|(stat['grp']==grps[i+1])]
            piv = tmp.pivot(index = 'grp', columns = 'label', values = tgt).T
            chis += [stats.chi2_contingency(piv)[1]]

        return chis

    def _llt_cap_func(self, x, s, b):
        """
        极值的处理函数，s,b 代表具体的数值
        """
        return max(s, min(x, b))

    def _bins_merge_chiq(self, tgt, cuts):
        """
        基于卡方的单调性保证：
        1.当分组的单调性不满足的时候，优先合并chiq检验最不显著的分组；
        2.重复以上步骤
        3.cuts要保持单调
        参数含义：
        1.tgt，目标特征及label
        2.cuts，对应bins
        """
        df = tgt.copy().dropna()
        ft_name, _ = df.columns.values

        df.loc[:, 'grp'] = pd.cut(df[ft_name], bins = cuts, labels = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(df)

        #单调性检验，同时合并卡方最不显著的分组
        while len(set([chis[i] < chis[i+1] for i in range(len(chis)-2)]))>1:
            lct = chis.index(max(chis))
            cuts.remove(cuts[lct+1])
            df = df.assign(grp = pd.cut(df[ft_name], bins = cuts, labels = range(len(cuts)-1), right = False))
            chis = self._chi_cal_func(df)

        return cuts

    def _setStrValue(self, strs2orders, ifraise = False):
        """
        给予字符串变量可比较性；
        strs2orders : 一个字典包含对应的字符串与数值对应关系
        """
        self.raw = self.raw.assign(**{self.ft_name:self.raw[self.ft_name].apply(lambda x: strs2orders[x] if x in strs2orders.keys() else None)})
        if ifraise:
            if self.raw[self.ft_name].isna().max() == True:
                raise ValueError('setStrValue: new value happened!')


    def _smpSizeCheck_real(self, tmp, prm_cuts, smp_size):
        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = prm_cuts, labels = range(len(prm_cuts)-1), right = False)
        stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()

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

            tmp = tmp.assign(grp=pd.cut(tmp[self.ft_name], bins = prm_cuts, labels = range(len(prm_cuts)-1), right = False))
            stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()

        return prm_cuts

    def eq_bins_func(self, data, tgt, grps = 2):
        pass

    def qlt_bins_func(self):
        """
        对于定性类的特征进行分箱统计
        对于无序的特征进行WOE编码有可能产生过拟合的可能，并不建议直接使用
        反馈的self.bins为一个dictionary，与其余反馈的标签不同
        """
        tmp = self.raw.copy().dropna()
        smp_size = np.int(len(tmp)*self.argms['pct_size'])+1

        values = list(set(list(tmp[self.ft_name].values)))

        strBins = {}
        for i in range(len(values)):
            strBins[values[i]] = i

        tmp['grp'] = tmp[self.ft_name].apply(lambda x: strBins[x])
        check = tmp.groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        check.columns = [self.ft_name, 'bad', 'size']
        check['bad_pct'] = check['bad']/check['size']

        check.sort_values(by = 'bad_pct', inplace = True)
        check.reset_index(drop = True, inplace = True)
        #return check, strBins

        while check['size'].min() < smp_size:
            loc = check['size'].idxmin()
#            if loc == len(check) - 1:
#                strBins[check.loc[loc-1, self.ft_name]] = strBins[check.loc[loc, self.ft_name]]
#            elif loc == 0:
#                strBins[check.loc[loc, self.ft_name]] = strBins[check.loc[loc+1, self.ft_name]]
#            elif check.loc[i-1, 'size'] < check.loc[i+1, 'size']:
#                strBins[check.loc[loc-1, self.ft_name]] = strBins[check.loc[loc, self.ft_name]]
#            else:
#                strBins[check.loc[loc, self.ft_name]] = strBins[check.loc[loc+1, self.ft_name]]

            if loc == len(check) - 1:
                repl = check.loc[loc-1, self.ft_name]
                tgt = check.loc[loc, self.ft_name]
            elif loc == 0:
                repl = check.loc[loc, self.ft_name]
                tgt = check.loc[loc+1, self.ft_name]
            elif check.loc[loc-1, 'size'] < check.loc[loc+1, 'size']:
                repl = check.loc[loc-1, self.ft_name]
                tgt = check.loc[loc, self.ft_name]
            else:
                repl = check.loc[loc, self.ft_name]
                tgt = check.loc[loc+1, self.ft_name]

            for k,v in strBins.items():
                if v == repl:
                    strBins[k] = tgt

            #return strBins
            #tmp['grp'] = check[self.ft_name].apply(lambda x: strBins[x])
            tmp = tmp.assign(grp = tmp[self.ft_name].apply(lambda x: strBins[x]))
            check = tmp.groupby('grp', as_index = False).agg({'label':['sum', 'count']})
            check.columns = [self.ft_name, 'bad', 'size']
            check['bad_pct'] = check['bad']/check['size']

            check.sort_values(by = 'bad_pct', inplace = True)
            check.reset_index(drop = True, inplace = True)


        self.bins = {self.ft_name: strBins}
        #实际上对于字符串型变量，添加最大值与最小值的意义不是特别明显
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}
        if len(set([v for k,v in strBins.items()])) == 1:
            self.woe_check = {self.ft_name: 'qlt_bins_func_failed!-value biased'}
        else:
            self.woe_check = {}

    def tree_bins_func(self, grps = None, pct_size = None):
        """
        基于决策树（信息熵）的分组
        1.max_grps控制最大分组的个数；
        2.pct_size控制每组最低的样本占比
        """
        tmp = self.raw.copy().dropna()
        if pct_size is None:
            smp_size = np.int(len(tmp)*self.argms['pct_size'])+1
        else:
            smp_size = np.int(len(tmp)*pct_size)+1
        if grps is None:
            grps = self.argms['max_grps']

        #当特征的最大取值占比超过阈值时，不做进一步区分，只分为2组
        #以决策树为分组的基准工具
        clf = DecisionTreeClassifier(min_samples_leaf = smp_size, max_leaf_nodes = grps)
        clf.fit(tmp[[self.ft_name]], tmp['label'])

        tmp['grp_prd'] = clf.apply(tmp[[self.ft_name]])

        grp_info = tmp.groupby('grp_prd').min()
        grp_info.sort_values(self.ft_name, inplace = True, ascending = True)
        cuts = list(grp_info[self.ft_name]) + [tmp[self.ft_name].max()+1]

        cuts = self._smpSizeCheck_real(tmp, cuts, smp_size)

        self.bins = {self.ft_name:cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}
        if len(cuts) == 2:
            self.woe_check = {self.ft_name: 'tree_bins_func_failed!-value biased'}
        else:
            self.woe_check = {}

    def _dist_bins_funcs(self, tgt, grps = 10):
        """
        基于特征取值范围进行分组，需要应对分布不均匀的情况
        目前看上去没有必要保留此函数
        """
        bins = [tgt.min() + i * np.float((tgt.max()-tgt.min()))/grps for i in range(grps)] + [tgt.max()]
        return bins

    def freq_bins_func(self, grps = None, pct_size = None):
        """
        基于频率的分组方式：
        1.grps控制分组的个数；
        2.pct_size控制任意分组的最小总体样本占比
        """
        if grps is None:
            grps = self.argms['max_grps']
        if pct_size is None:
            pct_size = self.argms['pct_size']

        tmp = self.raw.copy().dropna()
        #在已知分组数的前提下，允许任意分组样本占总体样本向下浮动一定的比例
        pct_size = min(pct_size, 1.0/grps/1.5)
        smp_size = np.int(len(tmp)*pct_size)+1

        prm_cuts = [tmp[self.ft_name].quantile(1.0/grps * a, interpolation = 'lower') for a in range(grps)]
        prm_cuts += [tmp[self.ft_name].max()+1]

        prm_cuts = list(set(prm_cuts))
        prm_cuts.sort()

        prm_cuts = self._smpSizeCheck_real(tmp, prm_cuts, smp_size)
        # tmp['grp'] = pd.cut(tmp[self.ft_name], bins = prm_cuts, index = range(len(prm_cuts)-1), right = False)
        # stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()
        #
        # rlts = []
        #
        # #最后检查分组后分组样本的占比情况，当样本个数欧过少时合并
        # while stat[self.ft_name].min() < smp_size:
        #     tgt_loc = np.argmin(stat[self.ft_name])
        #     if tgt_loc == 0:
        #         prm_cuts.remove(prm_cuts[tgt_loc+1])
        #     elif tgt_loc == len(prm_cuts)-2:
        #         prm_cuts.remove(prm_cuts[tgt_loc-1])
        #     elif stat.loc[tgt_loc-1,self.ft_name] < stat.loc[tgt_loc+1, self.ft_name]:
        #         prm_cuts.remove(prm_cuts[tgt_loc])
        #     else:
        #         prm_cuts.remove(prm_cuts[tgt_loc+1])
        #
        #     tmp['grp'] = pd.cut(tmp[self.ft_name], bins = prm_cuts, index = range(len(prm_cuts)-1), right = False)
        #     stat = tmp[['grp', self.ft_name]].groupby('grp', as_index = True).count()

        self.bins = {self.ft_name:prm_cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}
        if len(prm_cuts) == 2:
            self.woe_check = {self.ft_name: 'freq_bins_func_failed!-value biased'}
        else:
            self.woe_check = {}


    def chiq_bins_func(self, grps = None, pct_size = None):
        """
        通过chiq进行的分箱，先用频率分箱的方式分出较多组，后续迎chiq的方式进行合并
        1.grps控制初始频率分箱的组数；
        2.cuts可以指定初始分箱方式；
        3.pct_size控制最小分组样本占整体的最小比例；
        4.pv控制是否合并分箱的阈值
        """
        tmp = self.raw.copy().dropna()
        if grps is None:
            grps = self.argms['max_grps']
        pv = self.argms['chiq_pv']
        if pct_size is None:
            smp_size = np.int(len(tmp)*self.argms['pct_size'])+1
        else:
            smp_size = np.int(len(tmp)*pct_size)+1

        #先使用均匀分组当方式分割为20组
        self.freq_bins_func(grps = 20, pct_size = 0.03)

        cuts = self.getWoeBins()
        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = cuts, labels = range(len(cuts)-1), right = False)
        chis = self._chi_cal_func(tmp)
        if len(cuts) > 2:
        #以卡方为基准进行分箱合并
            while max(chis) > pv or len(cuts)-1>grps:
                if len(cuts) <= 3:
                    break
                tgt = chis.index(max(chis))
                cuts.remove(cuts[tgt+1])
                #tmp['grp'] = pd.cut(tmp[self.ft_name], bins = cuts, index = range(len(cuts)-1), right = False)
                tmp = tmp.assign(grp = pd.cut(tmp[self.ft_name], bins = cuts, labels = range(len(cuts)-1), right = False))
                chis = self._chi_cal_func(tmp)

            cuts = self._smpSizeCheck_real(tmp, cuts, smp_size)


        self.bins = {self.ft_name:cuts}
        self.cap_info = {'max': tmp[self.ft_name].max(), 'min':tmp[self.ft_name].min()}
        if len(cuts) == 2:
            self.woe_check = {self.ft_name: 'chiq_bins_func_failed!-value biased'}
        else:
            self.woe_check = {}

    def mono_bins_func(self):
        """
        检验任意分组的单调性：
        1.用卡方的方式确保单调性
        """
        bins = self.bins[self.ft_name]

        tmp = self.raw.dropna()

        #tmp.loc[:,self.ft_name] = tmp[self.ft_name].apply(self._llt_cap_func, s = self.cap_info['min'], b = self.cap_info['max'])
        tmp = tmp.assign(**{self.ft_name: tmp[self.ft_name].apply(self._llt_cap_func, s = self.cap_info['min'], b = self.cap_info['max'])})
        cuts = self._bins_merge_chiq(tmp, bins)
        self.bins = {self.ft_name:cuts}
        if len(cuts) == 2:
            self.woe_check = {self.ft_name: 'mono_bins_func_failed!-value biased'}
        else:
            self.woe_check = {}

    def getWoeBins(self):
        return self.bins[self.ft_name]

    def setWoeBins(self, bins):
        self.bins = bins

    def getWoeCheck(self):
        return self.woe_check.get(self.ft_name)

class WoeFuncs(bins_method_funcs):
    """docstring for WoeFuncs.
    具体使用方式：
    1.初始化类，定义分组最小占比，最大分组数，卡方合并阈值，是否单调，是否保留空值，分箱方式
    2.设定目标特征，提供特征及对应的label--class.setTgt()
    3.根据是否是数值型变量，使用class.woe_cal() 或者 class.strWoe_cal()
    """
    def __init__(self, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, keepnan = True, methods = 'tree'):
        super(WoeFuncs, self).__init__(pct_size, max_grps, chiq_pv)
        self.ifmono = ifmono; self.keepnan = keepnan
        self.methods = methods

        self.allInvalid = {}
        self.woeDetail = {}

    def setTgt(self, df):
        """
        df:必须是一个两列的DataFrame，其中第一列是特征，第二列是label
        """
        self.raw = df
        self.ft_name = list(self.raw.columns.values)[0]
        try:
            self.woeDetail[self.ft_name]['bins']
        except:
            self.woeDetail[self.ft_name] = {}

    def woe_cal(self, data = None):
        """
        计算特征的IV值及相应分箱的WOE编码
        1.data：意味着可以传输外部数据进行计算；
        2.ifiv: 控制是否返回最终的IV值；
        3.keepnan: 控制是否进行空值处理；
        4.methods: 控制分箱方法，只有‘tree’， ‘chiq’， ‘freq’三种可选；
        """
        methods = self.methods
        keepnan = self.keepnan

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

        if self.ifmono:
            self.mono_bins_func()

        bins = self.bins[self.ft_name]
        cap_info = self.cap_info

        tmp = data.dropna()
        #确保test及其他不会出现train上没有见到过当奇异值
        #tmp[self.ft_name] = tmp[self.ft_name].apply(self._llt_cap_func, s = cap_info['min'], b = cap_info['max'])
        tmp = tmp.assign(**{self.ft_name:tmp[self.ft_name].apply(self._llt_cap_func, s = cap_info['min'], b = cap_info['max'])})
        tmp['grp'] = pd.cut(tmp[self.ft_name], bins = bins, right = False).apply(str)
        stat = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        #是否保留空值当统计
        if keepnan and len(data[data[self.ft_name].isna()])>0:
            rlts = np.array(stat).tolist() + [['nan', data[data[self.ft_name].isna()]['label'].sum(), len(data[data[self.ft_name].isna()])]]
        else:
            rlts = np.array(stat).tolist()
        woe = pd.DataFrame(rlts, columns = [self.ft_name, 'bad', 'size'])

        #IV及woe值的计算
        bad = woe['bad'].sum(); good = woe['size'].sum() - bad
        woe['good'] = woe['size'] - woe['bad']
        woe['woe'] = ((woe['bad']/bad)/(woe['good']/good)).apply(np.log)
        woe['iv'] = (woe['bad']/bad - woe['good']/good) * woe['woe']
        woe['bad_pct'] = woe['bad']/woe['size']

        tmp_dict = {}
        for i in woe[self.ft_name].values:
            tmp_dict[i] = woe[woe[self.ft_name]==i].iloc[0]['woe']
        self.allInvalid[self.ft_name] = self.getWoeCheck()
        self.woeDetail[self.ft_name]['bins'] = self.getWoeBins()
        self.woeDetail[self.ft_name]['woes'] = tmp_dict

        self.iv_stat = {}
        self.iv_stat['woe_info'] = woe
        self.iv_stat['iv_value'] = woe['iv'].sum()

    def strWoe_cal(self, data = None, cuts = None, keepnan = True):
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
        cuts = self._strWoe_merge(tmp.dropna(), self.ft_name, cuts, self.argms['max_grps'])
        self.setWoeBins({self.ft_name:cuts})

        tmp['grp'] = tmp[self.ft_name].apply(lambda x: cuts[x] if x in cuts.keys() else 'nan')
        if not keepnan:
            tmp = tmp[tmp['grp']!='nan']

        woe = tmp[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        woe.columns = [self.ft_name, 'bad', 'size']
        bad = woe['bad'].sum(); good = woe['size'].sum() - bad
        woe['good'] = woe['size'] - woe['bad']
        woe['woe'] = ((woe['bad']/bad)/(woe['good']/good)).apply(np.log)
        woe['iv'] = (woe['bad']/bad - woe['good']/good) * woe['woe']
        woe['bad_pct'] = woe['bad']/woe['size']

        #return woe

        tmp_dict = {}
        for k, v in cuts.items():
            tmp_dict[k] = list(woe[woe[self.ft_name]==v]['woe'])[0]
            
        if 'nan' in list(woe[self.ft_name]):
            tmp_dict['nan'] = list(woe[woe[self.ft_name]=='nan'])[0]

        self.woeDetail[self.ft_name]['bins'] = self.getWoeBins()
        self.woeDetail[self.ft_name]['woes'] = tmp_dict
        self.allInvalid[self.ft_name] = self.getWoeCheck()

        self.iv_stat = {}
        self.iv_stat['woe_info'] = woe
        self.iv_stat['iv_value'] = woe['iv'].sum()

    def _strWoe_merge(self, df, ft_name, org_bins, max_grps):
        """
        对于分类型特征进行基于iv值的合并，需要提供特征名称，对应的原始分组，最大分组数
        同时需要根据最小分组的占比
        """
        df['grp'] = df[ft_name].apply(lambda x: org_bins[x] if x in org_bins.keys() else None)
        df = df[~df['grp'].isna()]

        woe = df[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
        woe.columns = [ft_name, 'bad', 'size']
        woe['bad_pct'] = woe['bad']/woe['size']
        woe.sort_values(by = 'bad_pct', inplace = True)
        woe.reset_index(drop=True,inplace=True)

        while len(woe)>max_grps:
            loc = woe['bad_pct'].diff(periods = 1).loc[1:].apply(abs).idxmin()
            m1, m2 = woe.loc[loc-1, ft_name], woe.loc[loc, ft_name]

            for k,v in org_bins.items():
                if v == m1:
                    org_bins[k] = m2
            #df['grp'] = df[ft_name].apply(lambda x: org_bins[x] if x in org_bins.keys() else 'nan')
            df = df.assign(grp = df[ft_name].apply(lambda x: org_bins[x] if x in org_bins.keys() else 'nan'))
            woe = df[['grp', 'label']].groupby('grp', as_index = False).agg({'label':['sum', 'count']})
            woe.columns = [ft_name, 'bad', 'size']
            woe['bad_pct'] = woe['bad']/woe['size']
            woe.sort_values(by = 'bad_pct', inplace = True)
            woe.reset_index(drop=True,inplace=True)

        return org_bins

    def woe_apply(self, data = None, keepnan = False):
        """
        用WOE编码替换分组信息
        """
        if data is None:
            data = self.raw.copy()

        if not keepnan:
            data = data.dropna()

        try:
            cuts = self.woeDetail[self.ft_name]['bins']
            woe_info = self.woeDetail[self.ft_name]['woes']
        except:
            self.woe_cal()
            cuts = self.woeDetail[self.ft_name]['bins']
            woe_info = self.woeDetail[self.ft_name]['woes']

        up, floor = max(cuts), min(cuts)

        data['grp'] = pd.cut(data[self.ft_name].apply(self._llt_cap_func, s = floor, b = up), bins = cuts, right = False).astype(object).apply(str)
        data = data.assign(grp = data['grp'].fillna('nan'))
        try:
            data[self.ft_name] = data['grp'].apply(lambda x: woe_info[x])
        except KeyError:
            raise KeyError('nan value happened in test!')

        self.woe_coded_data = data

    def strWoe_apply(self, data = None, keepnan = False):
        """
        对于定性类特征进行WOE编码替换
        """
        if data is None:
            data = self.raw.copy()

        if keepnan:
            data = data.dropna()

        try:
            cuts = self.woeDetail[self.ft_name]['bins']
            woe_info = self.woeDetail[self.ft_name]['woes']
        except:
            self.strWoe_cal()
            cuts = self.woeDetail[self.ft_name]['bins']
            woe_info = self.woeDetail[self.ft_name]['woes']

        try:
            data = data.assign(**{self.ft_name:data[self.ft_name].apply(lambda x: woe_info[x] if x in cuts.keys() else woe_info['nan'])})
        except KeyError:
            raise KeyError('nan value happened in test!')

        self.woe_coded_data = data

    def getWoeCode(self):
        return self.woe_coded_data[[self.ft_name]]

    def getWoeInfo(self):
        return self.iv_stat['woe_info']

    def getIVinfo(self):
        return self.iv_stat['iv_value']

    def getInvalid(self):
        return {k:v for k, v in self.allInvalid.items() if v is not None}

    def freeMmy(self):
        try:
            del self.raw
        except:
            pass

        try:
            del self.woe_code_data
        except:
            pass

class AllWoeFuncs(WoeFuncs):
    """docstring for AllWoeFUncs:
       1.check setFtrs and setData functions before any calculation
    """
    def __init__(self, path, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, keepnan = True, methods = 'tree'):
        super(AllWoeFuncs, self).__init__(pct_size, max_grps, chiq_pv, ifmono, keepnan, methods)
        self.path = path

    def setData(self, data):
        self.data = data

    def setFtrs(self, ftrs):
        """
        this functions is used before any woe calculations
        """
        if len(set(self.woeDetail.keys()) - set(self.data.columns.values))>0:
            raise ValueError('Features Not Match !!')
        else:
            self.ftrs = ftrs

    def AllWoeCollects(self, woe_vrs_info):
        """
        collect calculated woe infos
        """
        if type(woe_vrs_info) is dict:
            self.woeDetail = woe_vrs_info
        else:
            self.woeDetail = tools.getJson(self.path+'/feature_process_methods/IVstat/woeDetail_'+woe_vrs_info+'.json')

    def AllWoeCals(self, vrs = None):
        """
        vrs for version control
        """
        ivs = {}
        try:
            with tqdm(self.ftrs.keys()) as t:
                for i in t:
                    self.setTgt(self.data[[i, 'label']])
                    if self.ftrs[i] == 'str':
                        self.strWoe_cal()
                    elif self.ftrs[i] in ['int', 'float']:
                        self.woe_cal()
                    elif isinstance(self.ftrs[i], dict):
                        self._setStrValue(self.ftrs[i], ifraise = False)
                        self.woeDetail[i]['str2orders'] = self.ftrs[i]
                        self.woe_cal()

                    ivs[i] = self.getIVinfo()
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        if vrs is not None:

            tools.putFile(self.path+'/feature_process_methods/IVstat','IVs_'+vrs+'.json', ivs)
            tools.putFile(self.path+'/feature_process_methods/IVstat','woeDetail_'+vrs+'.json', self.woeDetail)

    def AllWoeApl(self, ifsave = True):
        try:
            if len(set(self.ftrs.keys())-set(self.woeDetail.keys())) > 0:
                raise ValueError('Run func AllWoeCals first')
            else:
                pass
        except:
            raise ValueError('Run func AllWoeCals first')

        #参数至少要传输一个label，但并无实际意义
        if 'label' in self.data.columns.values:
            self.data_woe = self.data[['label']]
        else:
            self.data['label'] = 1
            self.data_woe = self.data[['label']]

        try:
            with tqdm(self.woeDetail.keys()) as t:
                for i in t:
                    #print(i)
                    strSet = self.woeDetail[i].get('str2orders')
                    self.setTgt(self.data[[i, 'label']])
                    if strSet is None:
                        if isinstance(self.woeDetail[i]['bins'], dict):
                            self.strWoe_apply()
                        else:
                            self.woe_apply()
                    else:
                        self._setStrValue(strSet, ifraise = False)
                        self.woe_apply()

                    self.data_woe = pd.merge(left = self.data_woe, right = self.getWoeCode(), left_index = True, right_index = True, how = 'left')
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        if 'label' not in self.data.columns.values:
            self.data_woe.drop('label', axis = 1, inplace = True)

        if ifsave:
            self.data_woe.to_csv(self.path+'/ftrs_woe_code.csv')

    def getMdfData(self):
        return self.data_woe
