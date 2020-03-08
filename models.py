
import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def load_20_data():
    data=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)
    return(data)

def processtext(data):
    #Count vectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    print(X_train_counts.shape)
    return(X_train_counts)

def vocabulary(data):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    return(count_vect.vocabulary_)