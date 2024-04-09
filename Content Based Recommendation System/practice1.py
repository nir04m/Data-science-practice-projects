# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:26:07 2024

@author: Oghale Enwa
"""

import numpy as np 
import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

raw_data = pd.read_csv("netflix_titles.csv")
data = raw_data.copy()

data.dropna(subset=['cast','title','description','listed_in'],inplace=True,axis=0)
data = data.reset_index(drop=True)
drop_data = data.copy()


data['listed_in'] = [re.sub(r'[^\w\s]', '', t) for t in data['listed_in']]
data['cast'] = [re.sub(',',' ',re.sub(' ','',t)) for t in data['cast']]
data['description'] = [re.sub(r'[^\w\s]', '', t) for t in data['description']]
data['title'] = [re.sub(r'[^\w\s]', '', t) for t in data['title']]

data["combined"] = data['listed_in'] + '  ' + data['cast'] + ' ' + data['title'] + ' ' + data['description']
data.drop(['listed_in','cast','description'],axis=1,inplace=True)

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data["combined"])
cosine_similarities = linear_kernel(matrix,matrix)
movie_title = data['title']
indices = pd.Series(data.index, index=data['title'])

#def content_recommender(title):
#    idx = indices[title]
#    sim_scores = list(enumerate(cosine_similarities[idx]))
#    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#    sim_scores = sim_scores[1:31]
#    movie_indices = [i[0] for i in sim_scores]
#    return movie_title.iloc[movie_indices]

def content_recommender(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return  movie_title.iloc[movie_indices]

title = 'The Crown'
suggestions = content_recommender(title)

suggestions_df = pd.DataFrame(data=suggestions)
suggestions_df.to_csv('suggestions_based_on_%s.csv'%title,index=False,header=False)