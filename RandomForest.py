import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

testdata = models.load_20_train_data()
data = models.load_20_test_data()

grid_search_params = {'clf__criterion':['gini','entropy'],
                      'clf__n_estimators':[100,200,400,800],
                      'clf__max_depth':[2,4,8,16,32,None]
                     }

plainRFPred = models.run_pipeline(data, model = RandomForestClassifier(), params = grid_search_params, gridsearch=None).predict(testdata.data)
bestRF = models.run_pipeline(data, model = RandomForestClassifier(), params = grid_search_params, gridsearch=True)
bestRFPred = bestRF.predict(testdata.data)

print("Plain Logistic Regression Test Accuracy: ", accuracy_score(plainRFPred, testdata.target))
print("Best Logistic Regression Test Accuracy: ", accuracy_score(bestRFPred, testdata.target))

print("Confusion Matrix for Plain Logistic Regression Test", confusion_matrix(plainRFPred,testdata.target))
print("Confusion Matrix for Best Logistic Regression Test", confusion_matrix(bestRFPred,testdata.target))