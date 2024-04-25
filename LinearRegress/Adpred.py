# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:42:27 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("advertising.csv")
df = raw_data.copy()


#fig, axs = plt.subplots(3, figsize = (5,20))
#plt1 = sns.boxplot(df['TV'], ax = axs[0])
#plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
#plt3 = sns.boxplot(df['Radio'], ax = axs[2])
# plt.tight_layout()
#sns.boxplot(df['Sales'])
#sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
#plt.xlabel('TV')
#plt.ylabel('Sales')
#plt.scatter(df.TV,df.Sales,color='red',marker='+')


import sklearn
import scipy.stats as stats
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(df[['TV']] ,df.Sales)

#plt.xlabel('TV')
#plt.ylabel('Sales')
#plt.scatter(df.TV,df.Sales,color='red',marker='+')
#plt.plot(df.TV,reg.predict(df[['TV']]),color = 'blue')


reeg =reg.predict([[44.5]])
#print('Sales Prediction By TV ads',reeg,)

reeg =reg.predict([[20.25]])
#print('Sales Prediction By TV ads',reeg,)

reg = linear_model.LinearRegression()
reg.fit(df[['TV','Radio','Newspaper']] ,df.Sales)

#plt.figure(figsize=(15, 15))
#plt.xlabel('TV')
#plt.ylabel('Sales')
#plt.scatter(df.TV,df.Sales,color='red',marker='+')
#plt.plot(df.TV,df.Radio,df.Newspaper,reg.predict(df[['TV','Radio','Newspaper']]),color = 'blue')

#reg = linear_model.LinearRegression()
#reg.fit(df[['TV','Radio','Newspaper']] ,df.Sales)

#reeg =reg.predict([[151.5,41.3,58.5]])
#print('Sales',reeg,)
#reeg =reg.predict([[15.5,40.55,55.23]])
#print('Sales',reeg,)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['TV','Radio','Newspaper']] ,df.Sales)

reeg =reg.predict([[151.5,41.3,58.5]])
#print('Sales',reeg,)
reeg =reg.predict([[15.5,40.55,55.23]])
#print('Sales',reeg,)

#print(reg.coef_)

##print(reg.intercept_)



