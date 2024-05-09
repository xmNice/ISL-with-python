import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import (RocCurveDisplay, classification_report, confusion_matrix)
from sklearn.model_selection import (KFold, GridSearchCV, train_test_split)

from statsmodels.datasets import get_rdataset
from ISLP import (load_data, confusion_table)
from ISLP.svm import plot as plot_svm

# I.support vector classifier with linear boundary
# I.1 For data non linear separable
rng=np.random.default_rng(1) 
# Random generator，parameter is seed
X=rng.standard_normal((50, 2))
X.shape
y=np.array([-1]*25+[1]*25)
y
# 25 -1, 25 1
X[y==1]+=1
X[-10:]
# Generate 2 columns of random X, belonging to 2 classes -1 and 1; We see they are not linear separable. 
figure=plt.subplots(figsize=(4,4))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='Set2')

svm_linear=SVC(C=10, kernel='linear')
svm_linear.fit(X, y)
fig,ax=plt.subplots(figsize=(4,4))
plot_svm(X, y, svm_linear, ax=ax)
svm_linear.coef_

svm_linear_small=SVC(C=0.1, kernel='linear')
svm_linear_small.fit(X, y)
fig,ax=plt.subplots(figsize=(4,4))
plot_svm(X, y, svm_linear_small, ax=ax)
# c is small, margin is wider more support vectors
svm_linear_small.coef_
# Hyper plan for 2 cases are different

# Search the best C
kfold=KFold(random_state=0, n_splits=5, shuffle=True)
grid=GridSearchCV(estimator=SVC(kernel='linear'), param_grid={'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}, cv=kfold, refit=True, scoring='accuracy')
grid.fit(X,y)
grid.best_params_
# The best parameter is 1
grid.cv_results_['mean_test_score']
# The best score is 0.74
best=grid.best_estimator_

X_test=rng.standard_normal((20, 2))
y_test=np.array([-1]*10+[1]*10)
X_test[y_test==1]+=1
pred=best.predict(X_test)
print(classification_report(y_test, pred))
# 0.70 on test set
print(confusion_matrix(y_test,pred))

# I.2 For data linear separable, with large C et small C
X[y==1]+=1.9
figure=plt.subplots(figsize=(4,4))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='Set2')
svm_linear2=SVC(C=1e5, kernel='linear')
svm_linear2.fit(X, y)
fig,ax=plt.subplots(figsize=(4,4))
plot_svm(X, y, svm_linear2, ax=ax)
pred2=svm_linear2.predict(X)
# Data is separated by 3 support vectors without misclassification
print(classification_report(y, pred2))
print(confusion_matrix(y,pred2))
svm_linear2.coef_

svm_linear2_small=SVC(C=0.1, kernel='linear')
svm_linear2_small.fit(X, y)
fig,ax=plt.subplots(figsize=(4,4))
plot_svm(X, y, svm_linear2_small, ax=ax)
pred2_small=svm_linear2_small.predict(X)
# Data is separated by 12 support vectors without misclassification  margin is larger, will be more rubust on test set
print(classification_report(y, pred2_small))
print(confusion_matrix(y,pred2_small))
svm_linear2_small.coef_

# II Support vector machine with non linear boundary  Gaussian RBF kernel
X1=rng.standard_normal((200,2))
X1[:100]+=2
X1[100:150]-=2
X1.shape
y1=np.array([1]*150+[2]*50)
figure=plt.subplots(figsize=(8,8))
sns.scatterplot(x=X1[:,0],y=X1[:,1], hue=y1, palette='Set2')

X1_train, X1_test, y1_train, y1_test=train_test_split(X1, y1, random_state=0, test_size=0.5)

svm_rbf=SVC(kernel='rbf', gamma=1, C=1).fit(X1_train, y1_train)
pred_train=svm_rbf.predict(X1_train)
print(classification_report(y1_train, pred_train))
# Train error is not 0
fig,ax=plt.subplots(figsize=(8,8))
plot_svm(X1_train, y1_train, svm_rbf, ax=ax)

svm_rbf1=SVC(kernel='rbf', gamma=1, C=1e5).fit(X1_train, y1_train)
pred_train1=svm_rbf1.predict(X1_train)
print(classification_report(y1_train, pred_train1))
fig,ax=plt.subplots(figsize=(8,8))
plot_svm(X1_train, y1_train, svm_rbf1, ax=ax)
# When C is very large, margin is narrow, training error=0 risque of overfitting

# Search the best C and gamma
grid1=GridSearchCV(estimator=SVC(kernel='rbf'),param_grid={'C': [ 0.1, 1, 0, 100, 1000], 'gamma':[0.5,1,2,3,4]}, cv=kfold, refit=True, scoring='accuracy')
grid1.fit(X1_train, y1_train)
grid1.best_params_   
# The best C is 1, the best gamma is 0.5
grid1.cv_results_['mean_test_score']
# The best score is 0.89
best1=grid1.best_estimator_
pred_test1=best1.predict(X1_test)
print(classification_report(y1_test, pred_test1))
confusion_table(pred_test1,y1_test)
# Test accuracy is 0.89

# Compare test and training ROC curves  
roc_curve=RocCurveDisplay.from_estimator(best1, X1_train, y1_train, name='Training ROC') 
roc_curve=RocCurveDisplay.from_estimator(best1, X1_test, y1_test, name='Test ROC') 

# Draw 2 figs together
fig, ax=plt.subplots()
for X, y, n in zip ((X1_train,X1_test),(y1_train, y1_test), ('Training ROC','Test ROC')):
    RocCurveDisplay.from_estimator(best1, X, y, name=n, ax=ax) 
    
# III SVM multiclasses
rng1=np.random.default_rng(1)
X1=rng1.standard_normal((200,2))
X1[:100]+=2
X1[100:150]-=2
y1=np.array([1]*150+[2]*50)
rng2=np.random.default_rng(123)
X=np.vstack([X1,rng2.standard_normal((50,2))])
X
y=np.hstack([y1,[0]*50])
y
X[y==0,1]+=2
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='Set2')

# OVO
svm_rbf3=SVC(kernel='rbf',C=10, gamma=1, decision_function_shape='ovo')
svm_rbf3.fit(X, y)
fig,ax=plt.subplots(figsize=(8,8))
plot_svm(X, y, svm_rbf3, ax=ax)
y_pred=svm_rbf3.predict(X)
print(classification_report(y, y_pred))

# Grid search
grid3=GridSearchCV(estimator=SVC(kernel='rbf',decision_function_shape='ovo'),param_grid={'C': [ 0.1, 1, 0, 10, 100, 1000], 'gamma':[0.5,1,2,3,4]}, cv=kfold, refit=True, scoring='accuracy')
grid3.fit(X, y)
grid3.best_params_   
# The best C is 10, the best gamma is 0.5
grid3.cv_results_['mean_test_score']
best3=grid3.best_estimator_
pred_best=best3.predict(X)
print(classification_report(y, pred_best))
confusion_table(pred_best,y)

# OVR
svm_rbf4=SVC(kernel='rbf',C=10, gamma=0.5, decision_function_shape='ovr')
svm_rbf4.fit(X, y)
fig,ax=plt.subplots(figsize=(8,8))
plot_svm(X, y, svm_rbf4, ax=ax)
y_pred=svm_rbf4.predict(X)
print(classification_report(y, y_pred))

# IV SVC on gene expression data
khan=load_data('Khan')
df=khan.copy()
type(df)
df.keys()

df['xtrain'].shape,df['ytrain'].shape
df['xtest'].shape,df['ytest'].shape
df['ytrain'].value_counts()

svc_gene=SVC(kernel='linear', C=10, decision_function_shape='ovo')
svc_gene.fit(df['xtrain'], df['ytrain'])
pred_train=svc_gene.predict(df['xtrain'])
print(classification_report(df['ytrain'],pred_train))
# No misclassification on training data
pred_test=svc_gene.predict(df['xtest'])
print(classification_report(df['ytest'],pred_test))
print(confusion_matrix(df['ytest'],pred_test))
# 2 misclassification on test data, 2 class3 are misclassified to class2