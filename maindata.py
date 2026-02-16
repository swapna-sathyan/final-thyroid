
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.style.use("seaborn-whitegrid")       
#import pandas_profiling as pp 

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")
 
df = pd.read_csv("Thyroid.csv")
print(df.head())

print(df.info())

print(df.isnull().sum())

df.describe().T


df.hist(figsize=(10, 10), bins=50, xlabelsize=5, ylabelsize=5);


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve

X = df.drop(["CLass1"], axis = 1)
y = df["CLass1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30, 
                                                    random_state = 42)
 
 



 
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 8, 
                                  min_samples_split = 2,
                                  n_estimators = 1000)

rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


#print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

#print("Accuracy RF:",accuracy_score(y_test, y_pred))



rounded_accuracy = round(accuracy_score(y_test, y_pred), 2)   


print(f"Accuracy of Random Forest : {rounded_accuracy * 100:.2f}%")
print("****************************************")

feature_importances = rf_model.feature_importances_
plt.barh(X.columns, feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


 
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Predict the outcomes on the test set
y_pred = mlp.predict(X_test)

# Evaluate the model


accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

print(report)

rounded_accuracy1 = round(accuracy_score(y_test, y_pred), 2)   


print(f"Accuracy of Neural Network : {rounded_accuracy1 * 100:.2f}%")
print("****************************************")


 
