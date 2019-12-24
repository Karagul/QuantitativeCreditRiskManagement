import pandas as pd
import numpy as np

import re
import json
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import xgboost as xgb
import random
import warnings
from sklearn import metrics
from tqdm import tqdm

import model_builder 
import tools
from WoeMethods import bins_method_funcs

import sklearn
if sklearn.__version__ > '0.20.0':
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split


class ModelBasedMethods(object):
    """
    docstring for TreeBasedMethods:
    提供基于模型的feature选择函数
    随机选择feature并迭代，并根据模型的表现确定特征
    随机样本
    """
    def __init__(self, ftrs, label, features, corr, params, path, weights = None):
        self.ftrs = ftrs
        self.label = label
        if ftrs is not None:
            if len(set(list(features))-set(list(self.ftrs.columns.values))) >= 1:
                raise ValueError('feature N data not match!')
        self.corr = corr
        self.params = params
        self.path = path
        self.weights = weights
        self.features = list(ftrs.columns.values)

    def setFeatures(self, ftrs):
        """
        a list indicating the features used in this module, should be subset of self.ftrs.columns
        """
        self.features = ftrs

    def setCorr(self, corr):
        self.corr = corr

    def setParams(self, params):
        try:
            for i in params.keys():
                self.params[i] = params[i]
        except:
            raise ValueError('Invalid Parameters')

    def _random_select(self, ftrs, musthave = None, nums = 20, rnd_seed = None):
        #随机挑选特征
        if rnd_seed is not None:
            random.seed(rnd_seed)

        if musthave is not None:
            for i in musthave:
                ftrs.remove(i)

        tmp = random.sample(ftrs, nums)
        if len(tmp) < nums:
            warnings.warn('not enough features for random feature selections')
        return tmp

    def _random_select_cor(self, ftrs, nums, musthave = None, corr_c = 0.75, rnd_seed = None):
        #随机挑选特征，但要考虑相关性
        corr = self.corr.copy()
        if rnd_seed is not None:
            random.seed(rnd_seed)

        if musthave is not None:
            rlts = musthave
        else:
            rlts = []
        pcr = list(corr.index.values)

        for i in rlts:
            ops = corr.loc[pcr,i]
            pcr = list(ops[ops<corr_c].index.values)

        while  len(rlts)<nums and len(pcr)>0:
            rlts += [random.sample(pcr, 1)[0]]
            pc = corr.loc[pcr, rlts[-1]]
            pc = pc[pc<corr_c]
            pcr = list(pc.index.values)

        if len(rlts) < nums:
            warnings.warn('not enough features for random feature selections')
        return rlts

    def _model_perform_funcs(self, ylabel, ypred):
        #统计模型表现
        rlt = {}
        rlt['auc'] = metrics.roc_auc_score(ylabel, ypred)
        rlt['apr'] = metrics.average_precision_score(ylabel, ypred)
        rlt['logloss'] = metrics.log_loss(ylabel, ypred)
        return rlt

    def ftr_filter(self, tgt, tgt_c = 0.02, corr_c = 0.75):
        """
        根据特定指标及相关性进行特征选择
        tgt.shape = (n, 1)
        """
        corr = self.corr.copy()
        score_name = list(tgt.columns.values)[0]
        tgt.sort_values(score_name, ascending = False, inplace = True)
        tgt = tgt[score_name]
        tgt = tgt[tgt>tgt_c]
        lstd_bs = list(tgt.index.values)
        vld = []
        while len(lstd_bs) >= 1:
            vld += [lstd_bs[0]]
            lstd_bs = lstd_bs[1:]
            corr_check = corr[vld[-1]].loc[lstd_bs]
            corr_check = corr_check[corr_check>corr_c]
            invld = list(corr_check.index.values)

            for i in invld:
                lstd_bs.remove(i)

        return vld

    def _vif_filter(self, X, thres=10.0):
        """
        每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量
        理论上应该放到逻辑回归前面，暂时没有启用
        """
        col = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:,col].values, ix)
                   for ix in range(X.iloc[:,col].shape[1])]

            maxvif = max(vif)
            maxix = vif.index(maxvif)
            if maxvif > thres:
                del col[maxix]
                print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
                dropped = True
        print('Remain Variables:', list(X.columns[col]))
        print('VIF:', vif)
        return list(X.columns[col])

    def featureStat_model(self, ftrs = None, modeltype = 'lr', rnd_seed = None, test_size = 0.3):
        """
        根据model model_builder中的模型进行特征与模型表现的统计
        共有lrModel, xgbModel, cvModel三类
        """
        if ftrs is None:
            ftrs = self.features

        x = self.ftrs[ftrs]
        y = self.label
        if test_size == 0:
            train = x
            train_lb = y
            train_w = self.weights
            test = None
            test_lb = None
            test_w = None
            
        if self.weights is None:
            train, test, train_lb, test_lb = train_test_split(x, y, test_size = test_size, random_state = rnd_seed)
            train_w = None; test_w = None
        else:
            train, test, train_lb, test_lb, train_w, test_w = train_test_split(x, y, self.weights, test_size = test_size, random_state = rnd_seed)
        #return train, test, train_lb, test_lb 
        if modeltype == 'lr':
            if x.isnull().max().max() == 1:
                raise ValueError('unprocessed features with None values, pls check')

        if modeltype == 'lr':
            model = model_builder.lrModel(params = self.params)
        elif modeltype == 'xgb':
            model = model_builder.xgbModel(params = self.params)
        elif modeltype == 'cv':
            model = model_builder.cvModel(params = self.params)
        else:
            raise ValueError('other methods not provided')

        self.model_ = model.fit(train, train_lb, test, test_lb, train_w, test_w)
        self.model_perform_ = self.model_.getMperfrm()

    def getTvalues(self, mtrc, ifabs = False, name = None):
        feat_imp = pd.DataFrame(self.model_.getTvalues(mtrc), columns = ['fscore'])
        stat = feat_imp.to_dict()
        lfts = [a for a in self.model_.ft_names if a not in stat['fscore'].keys()]
        for i in lfts:
            stat['fscore'][i] = 0

        if name is not None:
            json_str = json.dumps(stat, indent= 4, ensure_ascii= False)
            with open(self.path + '/feat_imps/'+name+'_ftimp.json', 'w', encoding= 'utf-8') as f:
                f.write(json_str)
                f.close()

            with open(self.path + '/feat_imps/all_auc.json', 'a') as f:
                f.write(name+'_ftimp %.4f %.4f\n'%(self.model_perform_['train']['auc'], self.model_perform_['test']['auc']))
                f.close()
                
        rlt = pd.DataFrame(stat)
        if ifabs:
            rlt = rlt.assign(fscore = rlt['fscore'].apply(abs))

        return rlt

    def modelIprv_oneStep_plus(self, base_ftrs, tgts, modeltype = 'lr', rnd_seed = None, mtrc = 'auc', eval_s = 'train'):
        """
        可以用多进程优化
        """
        all_rlts = []
        for i in tgts:
            print(i)
            try:
                self.featureStat_model(base_ftrs+[i], modeltype, rnd_seed)
                all_rlts += [self.model_perform_[eval_s][mtrc]]
            except np.linalg.LinAlgError:
                all_rlts += [0]

        return {tgts[all_rlts.index(max(all_rlts))]:max(all_rlts)}

    def modelIprv_oneStep_minus(self, base_ftrs, modeltype = 'lr', rnd_seed = None, mtrc = 'auc', eval_s = 'train'):
        """
        可以用多进程优化
        """
        all_rlts = []
        for i in base_ftrs:
            ops = base_ftrs[:]
            ops.remove(i)
            try:
                self.featureStat_model(ops, modeltype, rnd_seed)
                all_rlts += [self.model_perform_[eval_s][mtrc]]
            except np.linalg.LinAlgError:
                all_rlts += [0]

        return {base_ftrs[all_rlts.index(max(all_rlts))]:max(all_rlts)}
    
    def featureSelection_randomSelect(self, ftr_names = None, modeltype = 'xgb', importance_type='gain',\
            threshold1 = 0.02,threshold2=0.01, threshold3=5, keep_rate=0.5, \
            max_iter=100, min_num = 20, test_size = 0.3):
        """
        one wx methods
        名义变量需提前处理
        仿照手动筛选变量过程,每次循环一部分变量被彻底删除而另一部分在下次循环保留
        threshold1:进入下一循环时gain百分占比大于该值的变量确定保留
        threshold2:当所有变量gain百分占比大于该值时停止循环    
        threshold3:必定淘汰的变量gain排名的倒数百分占比
        keep_rate不确定保留变量下次进入训练的概率
        threshold1 >= threshold2 >= threshold3
        """        
        #col保存每次剔除后剩余的所有变量
        if ftr_names is None:
            col = self.ftrs[self.features].columns
        else:
            col = self.ftrs[ftr_names].columns
        #subcol保存每次进入循环的变量
        subcol = col
        iter_num = 0
        while True:
            iter_num += 1
            self.featureStat_model(ftrs = subcol, modeltype = modeltype, rnd_seed = None, test_size = test_size)
            gain = pd.Series(data=np.zeros(len(subcol)),index=subcol)
            gain.loc[subcol] = gain.loc[subcol]+pd.Series(self.getTvalues(mtrc = importance_type)['fscore'])
            gain /= gain.sum()#计算带入列gain百分比
            gain.fillna(0,inplace=True)
            if_keep = gain > threshold2
            if if_keep.all():
                print("所有变量重要性占比均大于%g"%(threshold2))
                break
            elif iter_num >= max_iter or len(subcol[gain > np.percentile(gain,threshold3)])<=min_num:
                subcol = subcol[gain > np.percentile(gain,threshold3)]
                print("达到迭代次数或者最小变量数")
                break
            else:
                if_drop = gain <= np.percentile(gain,threshold3)
                if if_drop.any():
                    col = col.drop(subcol[if_drop],errors='ignore')
                keep_col = subcol[gain > threshold1]
                other_col = col.drop(keep_col,errors='ignore')
                subcol = keep_col.append( other_col[np.random.random_sample(len(other_col))<keep_rate])
            print("迭代:%d,%d个变量被保留,%d个变量确认被删除;\n最高分(%g) 出现在第%d轮"\
                  %(iter_num,len(subcol),if_drop.sum(),self.model_.model_.best_score,\
                    self.model_.model_.best_iteration))
        
        return subcol.tolist()
    
    def featureSelection_roundSelect(self, ftr_names = None, cycles = 12, modeltype = 'xgb', \
                                     step=4, importance_type=['gain','cover'], min_n=40, test_size = 0.3):
        """
        two wx methods
        X必须DataFrame
        名义变量需预处理
        step:每次剔除变量重要性后step%的变量
        输出保留变量(重要性从小到大排序)
        """
    
        def cal(list_=[]): 
            tmp="self.getTvalues(mtrc='"
            for count,i in enumerate(list_):
                tmp+="*self.getTvalues(mtrc='"+str(\
                    list_[count])+"')['fscore']" if count>0 else str(list_[count])+"')['fscore']"
            return tmp
        
        if ftr_names is None:
            columns = self.ftrs[self.features].columns
        else:
            columns = self.ftrs[ftr_names].columns

        gain=pd.Series(data=np.zeros(len(columns)),index=columns)
        try:
            with tqdm(range(cycles)) as t:
                for i in t:
                    self.featureStat_model(ftrs = columns, modeltype = modeltype, rnd_seed = None, test_size = test_size)
                    imp = eval(cal(list_ = importance_type))
                    gain.loc[imp.index] += imp
                    columns=imp[imp>np.percentile(imp,step)].index; gain =gain.loc[columns]
                    
                    if len(columns)<min_n:
                        break       
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        return columns[gain.argsort()[::-1]].tolist()

    def featureSelection_AvgScore(self, top = None, ftr_c = 0.65):
        """
        根据不断随机的抽取特征后的各模型表现进行特征评估
        """
        model_p = pd.read_table(self.path+'/feat_imps/all_auc.json', sep = ' ', names = ['files', 'train_auc', 'test_auc'])
        model_p = model_p[model_p['train_auc']>=ftr_c]
        vald_records = list(model_p['files'])

        for f in range(len(vald_records)):
            tgt = tools.getJson(self.path+'/feat_imps/'+vald_records[f]+'.json')
            rcd = pd.DataFrame(tgt)
            rcd.columns = [vald_records[f]]
            rcd.sort_values(vald_records[f], ascending = False, inplace = True)

            if top is not None:
                rcd = rcd.iloc[:top]
            if f == 0:
                rlt = rcd
            else:
                rlt = pd.merge(left = rlt, right = rcd, left_index = True, right_index = True, how = 'outer')
        
        if len(vald_records) == 0:
            raise ValueError('Not Enough Model')
        #return rlt
        fnl = pd.DataFrame(rlt.mean(axis = 1, skipna = True), columns = ['avg'])

        fnl['cnt'] = rlt.count(axis = 1, numeric_only = True)
        fnl['score'] = fnl['avg'] * fnl['cnt'].apply(lambda x: np.log(2+x))

        return fnl
