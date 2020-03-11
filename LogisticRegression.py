import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

testdata = models.load_20_train_data()
data = models.load_20_test_data()

grid_search_params ={'clf__C': [0.01, 0.05, 0.1, 0.3, 1],
                     'tfidf__use_idf': (True, False),
                     'clf__solver':['newton-cg', 'lbfgs', 'sag', 'saga'],
                     'clf__class_weight': ['balanced',None]
                    }

plainLRPred = models.run_pipeline(data, model = LogisticRegression(), params = grid_search_params, gridsearch=None).predict(testdata.data)
bestLR = models.run_pipeline(data, model = LogisticRegression(), params = grid_search_params,gridsearch=True)
bestLRPred = bestLR.predict(testdata.data)

print("Plain Logistic Regression Test Accuracy: ", accuracy_score(plainLRPred, testdata.target))
print("Best Logistic Regression Test Accuracy: ", accuracy_score(bestLRPred, testdata.target))

print("Confusion Matrix for Plain Logistic Regression Test", confusion_matrix(plainLRPred,testdata.target))
print("Confusion Matrix for Best Logistic Regression Test", confusion_matrix(bestLRPred,testdata.target))