import os
import sys
import json
import psutil
import platform
import pandas as pd
import numpy as np
from time import strftime, strptime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def getJson(file):
    with open(file, encoding= 'utf-8') as file:
        jsonStr = json.loads(file.read())
    return jsonStr

def getFiles(path):
    for root, dirs, files in os.walk(path):
        tgts = files[0]
        break
    return tgts

def putFile(path, file, jsonstr):
    if not os.path.exists(path):
        os.mkdir(path)
    json_str = json.dumps(jsonstr, indent= 4, ensure_ascii= False, cls=NpEncoder)
    with open(path+'/'+file, 'w', encoding= 'utf-8') as f:
        f.write(json_str)
        f.close()
