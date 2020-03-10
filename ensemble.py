import sklearn.ensemble.VotingClassifier
import pipeline

if __name__ == '__main__':
    test_ng = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))
    train_ng = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))

    pl = A2Pipeline()
    grid = pl.grid_search_tune()

    svc_clf = grid.fit(train_ng.data, train_ng.target)

    ensemble = VotingClassifier( estimators=[('svc', svc_clf), ('dt', dt_clf), ('lr', lr_clf)}])

    ensemble.fit(test_ng.data, test_ng.target)
    pred = ensemble.predict(train_ng)

    print(classification_report(test_ng.target, pred))
    print(np.meanpred == test_ng.target ))

