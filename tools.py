import os
import sys
import json
import psutil
import platform
import pandas as pd
from time import strftime, strptime

def getJson(file):
    with open(file, encoding= 'utf-8') as file:
        jsonStr = json.loads(file.read())
    return jsonStr

def getFiles(path):
    for root, dirs, files in os.walk(path):
        tgts = files[0]
        break
    return tgts

def putFile(path, jsonstr):
    json_str = json.dumps(ivs_new, indent= 4, ensure_ascii= False)
    with open(path, 'w', encoding= 'utf-8') as f:
        f.write(json_str)
        f.close()
