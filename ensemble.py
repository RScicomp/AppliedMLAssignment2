from sklearn.ensemble import VotingClassifier
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from pipeline import A2Pipeline
from sklearn.linear_model import LogisticRegression


def build_pipeline(model = DecisionTreeClassifier(random_state=0, criterion='gini')):
    return Pipeline(([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', model),
    ]))

if __name__ == '__main__':
    test_ng = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']), download_if_missing=True, categories=None, shuffle=True, random_state=42)
    train_ng = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']), download_if_missing=True, categories=None, shuffle=True, random_state=42)

    pl = A2Pipeline()

    svc_clf = pl.pipeline
    dt_clf = build_pipeline(model=DecisionTreeClassifier(splitter='random',class_weight='balanced',min_samples_leaf=1,min_samples_split=50,max_depth=200))
    lr_clf = build_pipeline(model=LogisticRegression())
    ensemble = VotingClassifier(voting='hard', weights=None,
            estimators=[('lr', lr_clf), ('svm', svc_clf)], n_jobs=-1)

    ensemble.fit(train_ng.data, train_ng.target)
    pred = ensemble.predict(test_ng.data)

    print(classification_report(test_ng.target, pred))
    print(np.mean(pred == test_ng.target ))
    print(np.std(pred == test_ng.target))

    train_pred = ensemble.predict(train_ng.data)

    print(classification_report(train_ng.target, train_pred))
    print(np.mean(train_pred == train_ng.target ))
    print(np.std(train_pred == train_ng.target))

