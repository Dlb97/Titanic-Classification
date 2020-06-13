# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:58:29 2020

@author: Usuario
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

X=train.drop(["Survived"],axis=1)
y=train["Survived"]
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=1,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=1,random_state=1)
adb_clf=AdaBoostClassifier(base_estimator=dt,n_estimators=100)
adb_clf.fit(X_train,y_train)
y_pred_proba=adb_clf.predict_proba(X_test)[:,1]
score=roc_auc_score(y_test,y_pred_proba)
import matplotlib.pyplot as plt
fpr,tpr,treshold=roc_curve(y_test,y_pred_proba)
plt.plot(fpr,tpr)