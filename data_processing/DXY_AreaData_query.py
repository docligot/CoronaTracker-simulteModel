# Query real time data from DingXiangYuan, and keep the latest records every day for each city

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:41:50 2020

@author: leebond

resource: https://github.com/jianxu305/nCov2019_analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import utils   # some convenient functions
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
from googletrans import Translator # package used to translate Chinese into English


translator = Translator()
## Resoruce for Chinese - English Translation
with open('chineseProvince_to_EN.pkl','rb') as f:
    prov_dict = pkl.load(f)
with open('chineseCity_to_EN.pkl','rb') as f:
    city_dict = pkl.load(f)
    # some new area is not included in city_dict, so manually update
    city_dict['密云区'] = 'MiYun District' 
    city_dict['武清区'] = 'WuQing District'
    city_dict['兵团第十二师'] = '12-th BingTuanGroup'
    city_dict['昌吉州'] = 'ChangJi'
    city_dict['阿克苏地区'] = 'AKeSu'
    

def getProvinceTranslation(name):
    return prov_dict[name]


def getCityTranslation(name):
    try:
        return city_dict[name]
    except KeyError:
        print(name + 'cannot be translated, ask Yiran to mannully Translate\n')

def translate_to_English(data, prov_dict, city_dict):
    """Translate Chinese in dataset to English
    """        
    data['provinceName'] = data['provinceName'].apply(getProvinceTranslation)
    data['cityName'] = data['cityName'].apply(getCityTranslation)
    return data
        
        
def main():
    
    DXYArea = utils.load_chinese_data() # Query latest Regional Data from DXY
    daily_frm_DXYArea = utils.aggDaily(DXYArea)  # aggregate to daily data (keep the latest records every day)
    
    ## Translate into English
    daily_frm_DXYArea = translate_to_English(daily_frm_DXYArea, prov_dict, city_dict)
    
    daily_frm_DXYArea.to_csv ('../data/DXYArea.csv', index = None, header=True)
    print("Save area daily dataset (English) into ../data/DXYArea.csv")
    
if __name__ == '__main__':
    main()