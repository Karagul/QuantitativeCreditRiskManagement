# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:11:19 2020

@author: zhuchang
"""

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

import tools
from WoeMethods import AllWoeFuncs, WoeFuncs
import FeatureStatTools
from FeatureProcess import putNaMethods


path = 'gt_big'
raw_data_file_name = 'raw_data.csv'

na_smy = tools.getJson(path+'/'+'na_value_info.json')
#na_smy = {
#          '':''}

raw = pd.read_csv(path+'/'+raw_data_file_name, header = 0)
for i in na_smy.keys():
    spurs = putNaMethods(na_list = [na_smy[i]])
    tmp = spurs.fit(raw[[i]]).transform(raw[[i]])
    raw = raw.assign(**{i:tmp[i]})
    
raw.to_csv(path+'/modify_data.csv', index = False)
