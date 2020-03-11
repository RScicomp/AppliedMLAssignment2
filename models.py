import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

import numpy as np

from time import time
import scipy.stats as stats
#from scipy.stats import loguniform


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def load_20_train_data():
    data=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(['headers', 'footers', 'quotes']), download_if_missing=True)
    return(data)

def load_20_test_data():
    data=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='test', categories=None, shuffle=True, random_state=42, remove=(['headers', 'footers', 'quotes']), download_if_missing=True)
    return data

def processtext(data):
    #Count vectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    print(X_train_counts.shape)
    
    #tfidf frequency transformation
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return(X_train_tf)

def vocabulary(data):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    return(count_vect.vocabulary_)

def cross_validate(model,X_input,target,cv):
    return(cross_val_score(model, X_input, target, cv=cv))

def loguniform(low=0, high=1, size=None):
    return stats.reciprocal(np.exp(low), np.exp(high))

def gethyperpars(model):
    return(model.get_params().keys())

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
#finding best hyperparameters
def randomizedsearch(X,y,model=DecisionTreeClassifier(random_state=0,criterion ='gini'),
                     n_iter_search=20,
                      param_dist = {'splitter':['best','random'],
                                    'max_features':["auto","sqrt","log2",None],
                                    'class_weight': ['balanced',None],
                                    'min_samples_leaf':[1,5,10,25,50,100],
                                    'min_samples_split':[2,5,10,25,50,100],
                                    'max_depth':[10,25,50,100,200]}):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search)
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
    return(randomized_search)

#finding best hyperparameters
def gridsearch(X,y,model=DecisionTreeClassifier(random_state=0,criterion ='gini'),
               param_grid = {'splitter':['best','random'],
                                    'max_features':["auto","sqrt","log2",None],
                                    'class_weight': ['balanced',None],
                                    'min_samples_leaf':[1,5,10,25,50,100],
                                    'min_samples_split':[2,5,10,25,50,100],
                                    'max_depth':[10,25,50,100,200]}):
    grid_search = GridSearchCV(model, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    return(grid_search)

# build pipeline with model  
def build_pipeline(model = LogisticRegression(random_state=0)):
    return Pipeline(([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',model),
    ]))

def run_pipeline(text_ds,model=LogisticRegression(random_state=0),
                 gridsearch = False,
                 params ={'clf__C': [0.01, 0.05, 0.1, 0.3, 1],
                          'tfidf__use_idf': (True, False),
                          'clf__solver':['newton-cg', 'lbfgs', 'sag', 'saga'],
                          'clf__class_weight': ['balanced',None]
                         }
                ):
    
    pl = build_pipeline(model)
    pl.fit(text_ds.data,text_ds.target)
    if(gridsearch != None):
        if(gridsearch==True):
            search = GridSearchCV(pl, params, n_jobs=-1,verbose=1)
        else:
            search =RandomizedSearchCV(pl, param_distributions=params,
                                       n_iter=10)
        start = time()
        search.fit(text_ds.data,text_ds.target)
        search.fit(data.data, data.target)
        print("SearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), 10))
        report(search.cv_results_)
        return(search)
    else:
        return(pl)