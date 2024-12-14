# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:04:10 2024

@author: Oghale Enwa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')


md_movies = pd.read_csv('tmdb_5000_movies.csv')
md_credits = pd.read_csv('tmdb_5000_credits.csv')

tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
md_movies['overview'] = md_movies['overview'].fillna("")
tfidf_matrix = tfidf.fit_transform(md_movies['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices  = pd.Series(md_movies.index, index=md_movies['original_title']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_sim):
    indx = indices[title]
    sim_scores = enumerate(cosine_sim[indx])
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_index = [i[0] for i in sim_scores]
    return md_movies['original_title'].iloc[sim_index]


print(get_recommendations('John Carter'))





































