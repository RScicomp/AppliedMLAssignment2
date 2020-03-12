import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time
import numpy as np


testdata=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='test', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)
data=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)

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
            
from sklearn.pipeline import Pipeline
def build_pipeline(model = DecisionTreeClassifier(random_state=0, criterion='gini')):
    return Pipeline(([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer(use_idf= True)),
        ('clf',model),
    ]))
def run_pipeline(text_ds,model=DecisionTreeClassifier(random_state=0, criterion='gini'),
                 gridsearch =False,
                 params ={'clf__splitter':['best','random'],
                          'tfidf__norm':['l1'],
                                    'clf__max_features':["auto","sqrt","log2",None],
                                    'clf__class_weight': ['balanced',None],
                                    'clf__min_samples_leaf':[1,5,10,20,50,100],
                                    'clf__min_samples_split':[2,10,20,50,100],
                                    'clf__max_depth':[10,25,50,100,None],
                                    'clf__max_leaf_nodes':[20,50,200,None]},
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
    
def traintest_models():
    #plain models
    plaindt = run_pipeline(data,DecisionTreeClassifier(random_state=0),gridsearch=None)
    plainada = run_pipeline(dataAdaBoostClassifier(DecisionTreeClassifier(max_depth=2),random_state=0),gridsearch=None)

    #Optimized
    bestdecitree=run_pipeline(data,gridsearch=True)
    bestada=run_pipeline(data,AdaBoostClassifier(random_state=0),
                          params={
                                 'clf__base_estimator':[DecisionTreeClassifier(max_depth=2)],
                                 'clf__n_estimators': [10,100,200,600],
                                 'clf__learning_rate':[.5,.9,1,1.5],
                                 'clf__algorithm':['SAMME.R','SAMME'],
                                 },gridsearch=True)
    
    
    #plain Predictions
    dtplainpred = plaindt.predict(testdata.data)
    adaplainpred = plainada.predict(testdata.data)

    #best Predictions
    adapred = bestada.predict(testdata.data)
    dtpred = bestdecitree.predict(testdata.data)

    print(accuracy_score(dtplainpred, testdata.target))
    print(accuracy_score(adaplainpred, testdata.target))
    print(accuracy_score(dtpred, testdata.target))
    print(accuracy_score(adapred, testdata.target))

    print(confusion_matrix(adaplainpred,testdata.target))
    print(confusion_matrix(dtplainpred,testdata.target))
    print(confusion_matrix(adapred,testdata.target))
    print(confusion_matrix(dtpred,testdata.target))

    return([plaindt,plainada,bestdecitree,bestada])