from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import glob, json, random

class A2Pipeline:

    def __init__(self, model=svm.LinearSVC(max_iter=1000, random_state=42)):
        self.model = model
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        return Pipeline(([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.model),
        ]))

    def fit_pipeline(self, text_ds):
        return self.pipeline.fit(text_ds.data, text_ds.target)

    def pred(self, estimator, test_x):
        return estimator.predict(test_x)

    def grid_search_tune(self):
        param_grid = {'clf__C': [0.1, 1, 10, 100]}

        return GridSearchCV(self.pipeline, param_grid, cv=3, verbose=True, n_jobs=-1)

    @staticmethod
    def load_imdb(sub_set='train'):
        path_pos = './imdb/aclImdb/{}/pos/'.format(sub_set)
        path_neg = './imdb/aclImdb/{}/neg/'.format(sub_set)
        
        neg_files = []
        pos_files = []

        for file in glob.glob(path_neg + "*.txt"):
            f = open(file, "r")
            neg_files.append(f.read())
        for file in glob.glob(path_pos + "*.txt"):
            f = open(file, "r")
            pos_files.append(f.read())

        neg = list(zip(neg_files, [0]*len(neg_files)))
        pos = list(zip(neg_files, [1]*len(pos_files)))
        sentiments = neg + pos

        random.shuffle(sentiments)

        dataset = {'data': [], 'target':[]}
        for s in sentiments:
            dataset['data'].append(s[0])
            dataset['target'].append(s[1])

        return (np.array(dataset['data']), np.array(dataset['target']))

if __name__ == '__main__':
    test_ng = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))
    train_ng = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))

    pl = A2Pipeline()
    baseline = A2Pipeline(svm.LinearSVC())
    grid = pl.grid_search_tune()

    tuned = grid.fit(train_ng.data, train_ng.target)
    baseline.fit_pipeline(train_ng)

    tuned_pred = pl.pred(tuned, test_ng.data)
    base_pred = baseline.pred(baseline.pipeline, test_ng.data) 

    print("tuned pred")
    print(classification_report(test_ng.target, tuned_pred))
    print(np.mean(tuned_pred == test_ng.target ))
    
    print((grid.cv_results_))

    print("base pred")
    print(classification_report(test_ng.target, base_pred))
    print(np.mean(base_pred == test_ng.target ))


    # IMDB
    imdb_train = A2Pipeline.load_imdb()
    imdb_test = A2Pipeline.load_imdb()

    i_baseline = A2Pipeline(svm.LinearSVC())
    i_pipeline = A2Pipeline()
    imdb_grid = i_pipeline.grid_search_tune()
    
    i_tuned = imdb_grid.fit(imdb_train[0], imdb_train[1])
    i_baseline.pipeline.fit(imdb_train[0], imdb_train[1])

    i_tuned_pred = i_pipeline.pred(i_tuned, imdb_test[0])
    i_base_pred = baseline.pred(baseline.pipeline,imdb_test[0])

    print('imdb tuned pred')
    print(classification_report(imdb_test[1], i_tuned_pred))
    print((imdb_grid.cv_results_))

    print("base imdb pred")
    print(classification_report(imdb_test[1], i_tuned_pred))

