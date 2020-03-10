
# coding: utf-8

# In[1]:


import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[2]:


testdata=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='test', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)
data=sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from time import time
import numpy as np
def load_data(data, input_features, target_feature, cat_features=None, num_features=None, txt_features=None):
    all_features = input_features + [target_feature]
    print(data.values.shape)
    data = data[all_features]
    data.dropna(subset=[target_feature], inplace=True)

    # change categorical features to numeric code
    if(cat_features!=None):
        data[cat_features] = data[cat_features].astype('category')
        data[cat_features] = data[cat_features].apply(lambda x: x.cat.codes)
    # replace nan with 0 in numerical features
    if(num_features!=None):
        data[num_features] = data[num_features].fillna(0.)
        for feature in num_features:
            data[feature] = data[feature].apply(lambda x: replace_string(x))
    if(txt_features!=None):
        if txt_features:
            data[txt_features] = data[txt_features].fillna('')

    return data
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
        ('tfidf',TfidfTransformer()),
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
        


# In[28]:


build_pipeline(model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),random_state=0)).get_params().keys()


# In[33]:


#plain models
plaindt = run_pipeline(data,gridsearch=None)
plainada = run_pipeline(data,gridsearch=None)


# In[ ]:


#Optimized
bestdecitree=run_pipeline(data,gridsearch=True)
bestada = run_pipeline(data,AdaBoostClassifier(random_state=0),
                      params={
                             'clf__base_estimators':[DecisionTreeClassifier(max_depth=2),DecisionTreeClassifier(max_depth=3)],
                             'clf__n_estimators': [10,50,100,200,600],
                             'clf__learning_rate':[.1,.3,.5,.7,.9,1,1.5],
                             'clf__algorithm':['SAMME.R','SAMME'],
                             },gridsearch=True)


# In[50]:


#best Predictions
adapred = bestada.predict(testdata.data)
dtpred = bestdecitree.predict(testdata.data)


# In[52]:


#plain Predictions
dtplainpred = plaindt.predict(testdata.data)
adaplainpred = plainada.predict(testdata.data)


# In[53]:


print(accuracy_score(dtplainpred, testdata.target))
print(accuracy_score(adaplainpred, testdata.target))
print(accuracy_score(dtpred, testdata.target))
print(accuracy_score(adapred, testdata.target))


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(adaplainpred,testdata.target))
print(confusion_matrix(dtplainpred,testdata.target))
print(confusion_matrix(adapred,testdata.target))
print(confusion_matrix(dtpred,testdata.target))


# In[7]:


#cuda
from sklearn.metrics import confusion_matrix, accuracy_score
def cross_validate(model,X_input,target,cv):
    return(cross_val_score(model, X_input, target, cv=cv))

cvada=cross_validate(bestada,testdata.data,testdata.target,3)
cvdecitree=cross_validate(bestdecitree,testdata.data,testdata.target,3)

print(cvada)
print(cvdecitree)

