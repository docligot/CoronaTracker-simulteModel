# Query real time data from DingXiangYuan, and keep the latest records every day for each city

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:41:50 2020

@author: leebond

resource: https://github.com/jianxu305/nCov2019_analysis
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import utils   # some convenient functions
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
from googletrans import Translator # package used to translate Chinese into English


translator = Translator()
## Resoruce for Chinese - English Translation
## Resoruce for Chinese - English Translation
with open('chineseProvince_to_EN.pkl','rb') as f:
    prov_dict = pkl.load(f)
with open('chineseCity_to_EN.pkl','rb') as f:
    city_dict = pkl.load(f)    
    

def isNaN(num):
    return num != num


        
def translate_to_English(data, prov_dict, city_dict):
    """Translate Chinese in dataset to English
    """        
    data['province'] = data['province'].apply(getProvinceTranslation)
    data['city'] = data['city'].apply(getCityTranslation)
    return data
    
def getProvinceTranslation(name):
    if not isNaN(name):
        return prov_dict[name]
    else: 
        return name

def getCityTranslation(name):
    try:
        if not isNaN(name) :
            return city_dict[name]
        else:
            return name
    except KeyError:
        if name != None and len(name)>0:
            print(name + 'cannot be translated, ask Yiran to mannully Translate\n')
        
        
def main():
    
    ## Query the latest data
    os.system('python dataset.py')
    
    DXYArea = pd.read_csv('../data/DXYArea.csv')
    # select column
    DXYArea = DXYArea[['date','country','countryCode','province', 'city', 'confirmed', 'suspected', 'cured', 'dead']]
    # clean data 
    DXYArea["city"] = DXYArea["city"].str.split('市',expand=True)[1].str[:-1]
    
    print("Save area daily dataset (English) into ../data/DXYArea.csv")
    
if __name__ == '__main__':
    main()