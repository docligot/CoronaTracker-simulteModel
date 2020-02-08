#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:41:50 2020

@author: leebond
"""

import pandas as pd
import pickle as pkl
from googletrans import Translator
translator = Translator()

dxy = pd.read_csv('./csv/DXYArea.csv')

prov = [translator.translate(province).text for province in dxy['provinceName'].unique().tolist()]
city = [translator.translate(province).text for province in dxy['cityName'].unique().tolist()]

prov_dict = dict(list(zip(dxy['provinceName'].unique().tolist(), prov)))
city_dict = dict(list(zip(dxy['cityName'].unique().tolist(), city)))


with open('chineseProvince_to_EN.pkl','wb') as f:
    pkl.dump(prov_dict, f)
with open('chineseCity_to_EN.pkl','wb') as f:
    pkl.dump(city_dict, f)