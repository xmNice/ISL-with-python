import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

from ISLP import load_data 

# I Load and understand the data
smarket=load_data('Smarket')
smarket.head()
smarket.info()
smarket.describe()
# only target is binary.
# Descriptive static values are normal and in the same order.
# No Na.

X=smarket.copy()
y=X.pop('Direction')== 'Up'
# y is transformed to boolean values

cor=X.corr()
cor
# No cor between lags.
# Intermediate positive cor bw year and volume, i.e volume increases along year.
X.plot(y='Volume')

# II Train models
# II.1 Logistic regression using lags and volume features
X=X.drop(['Year', 'Today'], axis=1)
X['Intercept']=np.ones(smarket.shape[0])
X.head()
model1=sm.GLM(y,X, family=sm.families.Binomial()).fit()
model1.summary()
# Take attention, for GLM(y, x), not GLM(x, y) as other functions.
# p-values are large. Lag1 has the smallest p, but 0.15 is still large. 
# Negative coef means that: it is less likely to go up today if yesterday is positive.

probability=model1.predict(X)
probability[:5]
labels=np.array(['Down']*1250)
labels[probability>0.5]='Up'
print(classification_report(labels, smarket.Direction, target_names = ['Up', 'Down']))
# print() makes the  output of classification_report() in table form without misaligned.
# Low accuracy, not surprising, because we see P values are not significant. 
# Remove some predictors, keep only lag1 and lag2, retrain the model. 
# Split data and test model with test set.

# Logistic regression using 2 lags
X=smarket.copy()
X.head()
X_train=X[X['Year']<2005].drop(['Year','Today','Lag3','Lag4','Lag5','Volume'], axis=1)
X_test=X[X['Year']>=2005].drop(['Year','Today','Lag3','Lag4','Lag5','Volume'], axis=1)
y_train,y_test=[m.pop('Direction') for m in [X_train, X_test]]
X_train['intercept']=np.ones(X_train.shape[0])
X_train.head()
y_train=y_train=='Up'
y_train
model2=sm.GLM(y_train,X_train, family=sm.families.Binomial()).fit()
model2.summary()
X_test['intercept']=np.ones(X_test.shape[0])
pred_test=model2.predict(X_test)
pred_test=np.where(pred_test>0.5, 'Up', "Down")
print(classification_report(y_test, pred_test, target_names = ['Up', 'Down']))

# II.2 Linear discriminant analysis
# LDA is a sklearn model, ade intercept automatically; train using only 2 lags
X=smarket.copy()
X.head()
X_train=X[X['Year']<2005].drop(['Year','Today','Lag3','Lag4','Lag5','Volume'], axis=1)
X_test=X[X['Year']>=2005].drop(['Year','Today','Lag3','Lag4','Lag5','Volume'], axis=1)
y_train,y_test=[m.pop('Direction') for m in [X_train, X_test]]

model3=LDA(store_covariance=True).fit(X_train, y_train)
model3.means_
# While default'solver' = 'svd', covariance is not computed, hence we need to set True.
model3.classes_
model3.priors_
# class ratio in training set
model3.scalings_
pred_test3=model3.predict(X_test)
pred_test3[:5]
print(classification_report(y_test, pred_test3, target_names = ['Up', 'Down']))
# result of logistic and LDA are nearly the same

# Compute probability by prediction method. We get probability predicted for each class. The class label is decided by default threshold p=0.5. We can change this threshold to justify output classification label.
model3.predict_proba(X_test)[:5]
model3.predict_proba(X_test)[:,1].max()
model3.predict_proba(X_test)[:,0].max()
# model3.classes_   to verify which column correspond to which class 

# II.3 Quadratic discriminant analysis using 2 lags
model4=QDA(store_covariance=True).fit(X_train,y_train)
model4.means_
model4.priors_
model4.covariance_
pred_test4=model4.predict(X_test)
pred_test4[:4]
print(classification_report(y_test, pred_test4, target_names = ['Up', 'Down']))
# Results are a little bit better than LR and LDA

# II.4 Gaussian Naive Bayes
model5=GaussianNB().fit(X_train, y_train)
pred_test5=model5.predict(X_test)
pred_test5[:5]
print(classification_report(y_test, pred_test5))
model5.theta_
model5.var_
model5.predict_proba(X_test)[:5]

# II.5 KNN N=3 is the best. For this dataset QDA performers the best 
knn1=KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
pred_testK1=knn1.predict(X_test)
pred_testK1[:5]
print(classification_report(y_test, pred_testK1))

knn3=KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
pred_testK3=knn3.predict(X_test)
pred_testK3[:5]
print(classification_report(y_test, pred_testK3))

knn5=KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred_testK5=knn5.predict(X_test)
pred_testK5[:5]
print(classification_report(y_test, pred_testK5))

# III Caravan dataset binary classification 
Caravan=load_data('Caravan')
Caravan.shape
Caravan.info()
Caravan.columns
Caravan.describe()
Caravan.Purchase.value_counts()

X=Caravan.copy()
y=X.pop('Purchase')
X_scaled=StandardScaler(with_mean=True, with_std=True,copy=True).fit_transform(X)
X_scaled[:1]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=1000)

# III.1 KNN
for k in range(1,6):
    knn=KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    pred_test=knn.predict(X_test)
    print(f'The result for k={k} is:')
    print(classification_report(y_test, pred_test))
# k=2 is the best precision=0.17  K=4 precision=0

# III.2 Logistic regression
lX_train=X_train.copy()
intercept_train=np.ones(X_train.shape[0]).reshape(-1,1)
lX_train=np.concatenate((lX_train, intercept_train),axis=1)
lX_train.shape

lX_test=X_test.copy()
intercept_test=np.ones(X_test.shape[0]).reshape(-1,1)
lX_test=np.concatenate((lX_test, intercept_test),axis=1)

ly_train=y_train== 'Yes'
ly_train[:5]

ly_test=y_test=='Yes'
ly_test[:5]

logis_model=sm.GLM(ly_train, lX_train, family=sm.families.Binomial()).fit()

logis_pred= logis_model.predict(lX_test)
logis_pred.shape
label_test=np.where(logis_pred>0.5, 'Yes','No')
label_test[:5]
print(classification_report(y_test, label_test))
# Use a cut-off=0.5  0 people will purchase

label_test2=np.where(logis_pred>0.25, 'Yes','No')
label_test2[:5]
print(classification_report(y_test, label_test2))
# Use a cut-off=0.25  precision =0.31, random guessing =0.06 about
# Classifier is 5 times better than random guess