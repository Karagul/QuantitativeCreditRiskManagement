import pandas as pd
import numpy as np

import os
import json
import re

from WoeMethods import *
from FeaturePreprocessUnspv import *
from tools import *

#特征的处理方法
class featureProcess(AllWoeFuncs, FeatureProcessFuncs):
    """docstring for ."""
    def __init__(self, pct_size = 0.05, max_grps = 5, chiq_pv = 0.05, ifmono = True, ifnan = True, methods = 'tree'):
        featurePreprocess.__init__(self)
        FeatureProcessFuncs.__init__(self, pct_size, max_grps, chiq_pv, ifmono, ifnan, methods)


    def setData(self, data):
        self.data = data

    def setFtrs(self, adict):
        """
        a very complicated dicionary, indicating specific processing methods for average single features is required here:
        1.No Processing;
        2.fillna with (mean, average, min, max, or any other value)；
        3.one hot;
        4.set quantitative values for str values;
        5.woe coding for (str or numbers)
        """

    def
