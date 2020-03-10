
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import numpy as np

from time import time
import scipy.stats as stats
#from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def build_pipeline(model = DecisionTreeClassifier(random_state=0, criterion='gini')):
    return Pipeline(([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer(use_idf: True)),
        ('clf',model),
    ]))

ada=build_pipeline(model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators = 600, learning_rate=1,algorithm='SAMME',random_state=0))
dt=build_pipeline(DecisionTreeClassifier(splitter='random',class_weight='balanced',min_samples_leaf=1,min_samples_split=50,max_depth=200))


ada.fit(data.data,data.target)
dt.fit(data.data,data.target)
