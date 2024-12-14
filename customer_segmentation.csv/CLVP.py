# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:20:00 2024

@author: Oghale Enwa
"""
from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb


tx_data = pd.read_csv('customer_segmentation.csv', encoding='cp1252')
raw_data = tx_data.copy()

#initate plotly
pyoff.init_notebook_mode()

#converting the type of Invoice Date Field from string to datetime
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

#creating YearMonth field for the ease of reporting and visualization
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)

tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
tx_canada = tx_data.query("Country=='Canada'").reset_index(drop=True)

#create a generic user dataframe to keep CustomerID and new segmentation score
tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
tx_user.columns = ['CustomerID']

#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

# Compare the last transaction of the dataset with last transaction dates of the individual customer IDs.
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')


from sklearn.cluster import KMeans
sse={} # error
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init="auto").fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")

#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4, n_init="auto")
tx_user['RecencyCluster'] = kmeans.fit_predict(tx_user[['Recency']])

tx_user.groupby('RecencyCluster')['Recency'].describe()
