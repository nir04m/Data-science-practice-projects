# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:41:33 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('car_prices.csv')
car_ds = raw_data.copy()
from dateutil import parser
def parse_dates(date):
    # Check if the date is a string
    if not isinstance(date, str):
        return None 
    try:
        return parser.parse(date)
    except ValueError:
        return None  


car_ds['saledate'] = car_ds['saledate'].apply(parse_dates)
car_ds = car_ds.drop(['vin'], axis=1)
car_ds = car_ds.drop(['interior'], axis=1)

#print(car_ds.isnull().values.any())
#print(car_ds.isnull().sum())





















































