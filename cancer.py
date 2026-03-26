from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
# dataset load
# x = feature 
# y = target
data = load_breast_cancer()
x = data.data
y = data.target

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

# model train
model = LogisticRegression(max_iter=50000)
model.fit(X_test, Y_test)

# prediction
y_pred = model.predict(X_test)

# accuracy check
accuracy = accuracy_score(Y_test, y_pred)
print("LogisticRegression :-")
print("Accuracy  : ", accuracy)
print("comfusion metrix :")
print(confusion_matrix(Y_test, y_pred))
print("precision : ", precision_score(Y_test, y_pred))
print("recall: ", recall_score(Y_test, y_pred))
print("F1_score :", f1_score(Y_test, y_pred))


# decision tree model

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
dt.acc = accuracy_score(Y_test, dt_pred)
print("Decision tree :- ")
print("accuracy : ", dt.acc)
print("confusion metrix : ")
print(confusion_matrix(Y_test, dt_pred))
print("precision  : ", precision_score(Y_test, dt_pred))
print("recall :",recall_score(Y_test, (dt_pred)))
print("f1_score :", f1_score(Y_test,y_pred))

# ---- random forest model----
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_pred)
print("Randomforest :-  ")
print("accuracy : ", rf_acc)
print("confusion metrix :")
print(confusion_matrix(Y_test, rf_pred))
print("precision : ", precision_score(Y_test, rf_pred))
print("recall : ", recall_score(Y_test, rf_pred))
print("f1_score : ", f1_score(Y_test, rf_pred))

#---------cross validation-----------------

scores = cross_val_score(model, x,y,cv = 5)
print("cross validation score : ", scores)
print("average score : ", scores.mean())


#--------------heatmap plot----------------

import seaborn as sns
import matplotlib.pyplot as plt

# cm = confusion_matrix(Y_test, y_pred)
# plt.figure(figsize=(5,4))

# sns.heatmap(cm, annot = True, fmt = "d", cmap="Blues")

# plt.xlabel("predicted")
# plt.ylabel("Actual")
# plt.title("Confusion metrix - LogisticRegression")
# plt.show()

# cm = confusion_matrix(Y_test, dt_pred)
# plt.figure(figsize=(5,4))

# sns.heatmap(cm, annot = True, fmt = "d", cmap="Blues")

# plt.xlabel("predicted")
# plt.ylabel("Actual")
# plt.title("Confusion metrix - DecisionTree")
# plt.show()

# cm = confusion_matrix(Y_test, rf_pred)
# plt.figure(figsize=(5,4))

# sns.heatmap(cm, annot = True, fmt = "d", cmap="Blues")

# plt.xlabel("predicted")
# plt.ylabel("Actual")
# plt.title("Confusion metrix - DecisionTree")
# plt.show()

# feature convert code---------------
import numpy as np
import pandas as pd
importance = rf.feature_importances_
feature_name = data.feature_names
feat_imp = pd.Series(importance,index = data.feature_names)
feat_imp.sort_values().plot(kind='barh')
plt.title("Feature importance - Randomforest")
plt.xlabel("ipmortance")
plt.show()