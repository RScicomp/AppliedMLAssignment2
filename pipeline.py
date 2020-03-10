from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

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


if __name__ == '__main__':
    test_ng = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))
    train_ng = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))

    pl = A2Pipeline()
    grid = pl.grid_search_tune()

    tuned = grid.fit(train_ng.data, train_ng.target)

    tuned_pred = pl.pred(tuned, test_ng.data)

    print("tuned pred")
    print(classification_report(test_ng.target, tuned_pred))
    print(np.mean(tuned_pred == test_ng.target ))






