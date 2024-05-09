import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

 from ISLP import load_data
 from ISLP.models import (sklearn_sm, ModelSpec as MS)

 import statsmodels.api as sm
 from sklearn.model_selection import (train_test_split, KFold, cross_validate, ShuffleSplit)
 from sklearn.preprocessing import PolynomialFeatures
 
 # I Load and understand data 
 auto=load_data('Auto')
 df=auto.copy()
 df.head()
 
 sns.displot(df['mpg'])
 sns.displot(df['horsepower'])
 sns.boxplot(df['horsepower'])
 
# II Train models
 df_train, df_test= train_test_split(df, random_state=0, test_size=0.5)
 X_train_lm=sm.add_constant(df_train['horsepower'])
 y_train_lm=df_train['mpg']
 X_test_lm=sm.add_constant(df_test['horsepower'])
 y_test_lm=df_test['mpg']
 X_test_lm.shape
 
 model1=sm.OLS(y_train_lm, X_train_lm).fit()
 
 model1.summary()
 mse_train=np.mean((y_train_lm-model1.fittedvalues)**2)
 mse
 y_test_pred=model1.predict(X_test_lm)
 mse_test=np.mean((y_test_lm-y_test_pred)**2)
 mse_test
 
 def model_val(X_train, y_train, X_test, y_test):
     model=sm.OLS(y_train, X_train).fit()
     print(model.summary())
     y_test_pred=model.predict(X_test)
     mse_test=np.mean((y_test-y_test_pred)**2)
     print(f'\n mse_test:{mse_test}')
 
for i in range(1,4):
    poly=PolynomialFeatures(degree=i)
    X_train=poly.fit_transform(df_train['horsepower'].values.reshape(-1,1))
    y_train=df_train['mpg']
    X_test= poly.fit_transform (df_test['horsepower'].values.reshape(-1,1))
    y_test=df_test['mpg']
    print(f'Results for poly{i}:')
    model_val(X_train, y_train, X_test, y_test)
    print('\n\n')
# Change series to array, the other way: a= np.array(df_test['horsepower'])

# sklearn_sm (sm.OLS) connect sklearn and sm model
# X_train_lm, y_train_lm  contains intercept columns
hp_model=sklearn_sm(sm.OLS)
cv_results= cross_validate(hp_model, X_train_lm, y_train_lm, cv=df_train.shape[0])
mse_loocv=np.mean(cv_results['test_score'])
mse_loocv

# Fit polynomial X degree 1 2 3 4 5 with LOOCV (leave one out cross validation)
mse=np.zeros(5)
for i in range(0,5):
    X_poly=np.power.outer(np.array(df_train['horsepower']), np.arange(i+2))
    cv_results2= cross_validate(hp_model, X_poly, y_train_lm, cv=df_train.shape[0])
    mse_loocv=np.mean(cv_results2['test_score'])
    mse[i]=mse_loocv
mse  

# np.power.outer(x, 2)  =x**0+x**1+x**2
ax=sns.lineplot(y=mse, x=np.arange(1,6),marker='o', markersize=10)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('MSE (LOOCV)')

# Fit polynomial X degree 1 2 3 4 5 with 10 folds
cv10=KFold(random_state=0, n_splits=10, shuffle=True)
mse_10fold=np.zeros(5)
for i in range(0,5):
    X_poly=np.power.outer(np.array(df_train['horsepower']), np.arange(i+2))
    cv_results= cross_validate(hp_model, X_poly, y_train_lm, cv=cv10)
    mse=np.mean(cv_results['test_score'])
    mse_10fold[i]=mse
mse_10fold
ax=sns.lineplot(y=mse_10fold, x=np.arange(1,6),marker='o', markersize=10)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('MSE (10_Fold)')