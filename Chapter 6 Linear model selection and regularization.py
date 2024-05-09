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
 from ISLP.models import (sklearn_selected, Stepwise, sklearn_selection_path)
 
 import statsmodels.api as sm
 from sklearn.model_selection import (KFold, cross_validate, cross_val_predict, ShuffleSplit, GridSearchCV)
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import (ElasticNet,ElasticNetCV,LinearRegression)
 from sklearn.pipeline import Pipeline
 from sklearn.decomposition import PCA
 from sklearn.cross_decomposition import PLSRegression

# I load and understand data
hitters=load_data('Hitters')
df=hitters.copy()
df.head()
df.info()
# 322*20; have NA in Salary feature; League, Division, Newleague are category features.
df.describe()
df.isna().sum()
df.dropna(inplace=True)
df.shape

# II Forward stepwise selection based on training mse and Cp to select feature subset 
# Cp= (RSS+2*d*sigmahat**2)/n, negative cp statistic. 
# sigmahat is estimated residual variance; d is number of predictors.

# Train a full model to estimate sigma**2 and inspect summary for full model
df=pd.get_dummies(df, drop_first=True)
df.head()
y=df.pop('Salary')
y.head()
sns.displot(y)

X=sm.add_constant(df)
X.columns
features_num=['AtBat', 'Hits', 'HmRun', 'Runs','RBI', 'Walks', 'Years','CAtBat', 'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks', 'PutOuts','Assists', 'Errors']
X[features_num]=StandardScaler().fit_transform(X[features_num])
X.head()
model_full=sm.OLS(y,X).fit()
model_full.summary()
model_full.scale
sigma2=model_full.scale

# Define a function to calculate negative nCp
def nCp(estimator, X, y):
    n, d=X.shape
    yhat=estimator.predict(X)
    RSS=np.sum((y-yhat)**2)
    return -((RSS+2*d*(sigma2))/n)

# SKlearn does not have stepwise selection, so use ISLP lib
hitters1=hitters.copy()
hitters1.dropna(inplace=True)
hitters1.shape
hitters1[features_num]=StandardScaler().fit_transform(hitters1[features_num])
design=MS(hitters1.columns.drop('Salary')).fit(hitters1)
design.terms
hitters1
y1=np.array(hitters1['Salary'])

strategy=Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms))

hitters1_MSE = sklearn_selected(sm.OLS,strategy)
hitters1_MSE.fit(hitters1, y1)
len(hitters1_MSE.selected_state_)
# Default selection criteria is mse, as a result, 20 features are selected. Because adding features decrease always mse.

hitters1_Cp = sklearn_selected(sm.OLS,strategy, scoring=nCp)
hitters1_Cp.fit(hitters1, y1)
hitters1_Cp.selected_state_
# Set CP as criteria, 10 features are selected. 
# Compared to full model, these 10 features have low p-values. But some of them have p-values> 0.05.

# Train a new model with new features and inspect p-values and adjusted-R.
new_features=['const','Assists','AtBat','CAtBat','CRBI','CRuns','CWalks','Division_W','Hits','PutOuts','Walks']
X_new= X[new_features]
X_new.head()
X_new.shape
model_new=sm.OLS(y,X_new).fit()
model_new.summary(hitters1, y1)

# III Forward stepwise, use cross_validation method and validation method to select stepwise model model
strategy=Stepwise.fixed_steps(design, len(design.terms), direction="forward")
full_path=sklearn_selection_path(sm.OLS, strategy)
full_path.fit(hitters1, y1)
yhat_in=full_path.predict(hitters1)
yhat_in.shape

# In total, there are 20 models. Compute mse for each model. 
insample_mse=((yhat_in-y1.reshape(-1,1))**2).mean(axis=0)
insample_mse
ax=sns.lineplot(y=insample_mse, x=np.arange(insample_mse.shape[0]), marker='o')
ax.set_xticks(np.arange(insample_mse.shape[0])[::1])
ax.set_xlabel('Steps')
ax.set_ylabel('MSE')

kfold=KFold(n_splits=5, random_state=0, shuffle=True)
yhat_cv=cross_val_predict(full_path, hitters1, y1, cv=kfold)
yhat_cv.shape

cv_mse_folds=[]
for train_idx, test_idx in kfold.split(y1):
    mse=((yhat_cv[test_idx]-y1[test_idx, None])**2).mean(axis=0)
    cv_mse_folds.append(mse)
np.array(cv_mse_folds).T.shape
# In total, there are 20 models. Each model has 5 mses (one mse for one fold). 
# Transform list to array

cv_mse_SE=np.array(cv_mse_folds).T.std(axis=1)/np.sqrt(5)
cv_mse_SE
cv_mse=np.array(cv_mse_folds).T.mean(axis=1)
cv_mse
# For each model, compute standard error of the mean for the 5 mses. SEM = σ/√n 
# Dunno how to add standard error bar in figure :(

# Plot 2 lines together
ax=sns.lineplot(y=insample_mse, x=np.arange(insample_mse.shape[0]), marker='o', label='Train MSE')
sns.lineplot (y=cv_mse, x=np.arange(insample_mse.shape[0]), marker='o', c='r', label='5_folds MSE')
ax.set_xticks(np.arange(insample_mse.shape[0])[::1])
ax.set_xlabel('Steps')
ax.set_ylabel('MSE')

# Use validation set to select model
validation=ShuffleSplit(n_splits=1, test_size=0.2,random_state=0)
for train_idx, test_idx in validation.split(y1):
    full_path.fit(hitters1.iloc[train_idx], y1[train_idx])
    yhat_val=full_path.predict(hitters1.iloc[test_idx])
    errors=(yhat_val-y1[test_idx, None])**2
    validation_mse=errors.mean(axis=0)
validation_mse

# Plot 3 lines together
ax=sns.lineplot(y=insample_mse, x=np.arange(insample_mse.shape[0]), marker='o', label='Train MSE')
sns.lineplot (y=cv_mse, x=np.arange(insample_mse.shape[0]), marker='o', c='r', label='5_folds MSE')
sns.lineplot (y=validation_mse, x=np.arange(insample_mse.shape[0]), marker='o', c='g', label='validation MSE')
ax.set_xticks(np.arange(insample_mse.shape[0])[::1])
ax.set_xlabel('Steps')
ax.set_ylabel('MSE')

# Based on ms, the models in the 6th and 7th step are the best.
# How to grab out these models？

# IV Ridge and lasso
df=pd.get_dummies(df, drop_first=True)
df.head()
y=df.pop('Salary')
y.head()
X=df
X_scaled=StandardScaler().fit_transform(X)
X_scaled[:3]
# scaled categorical and numeric features together.
y.std()
lambdas=10**np.linspace(8,-2,100)/y.std()
lambdas[:5]

# Coefficients are stored in column
lambda_path=ElasticNet.path(X_scaled, y, l1_ratio=0, alphas=lambdas)
type(lambda_path) 
# tuple [0] store lambda, [1] store coefficient; 19 rows because of 19 features; 100 columns because 100 lambdas.
lambda_path[1].shape

coef_path=pd.DataFrame(lambda_path[1].T, columns=df.columns, index=-np.log(alphas))
coef_path.index.name='-log(alpha)'
coef_path.head()

for i in coef_path.columns:
    ax=sns.lineplot(x=coef_path.index, y=coef_path[i], label=i)
    ax.set_ylabel('Standardized coefficients')
    ax.legend(fontsize=6,bbox_to_anchor= (1.2,1))

beta_hat39=coef_path.iloc[39]
lambda_path[0][39], beta_hat39
lambdas[39]
np.linalg.norm(beta_hat39)

beta_hat59=coef_path.iloc[59]
lambda_path[0][59], beta_hat59
lambdas[59]
np.linalg.norm(beta_hat59)
# lambdas is listed in descending order, lambda (index_59)<lambda (index_39), coefficient 59 > coefficient 39

# Use pipeline
scaler=StandardScaler(with_mean=True, with_std=True)
ridge=ElasticNet(alpha=lambdas[59], l1_ratio=0)
pipe=Pipeline(steps=[('scaler',scaler),('ridge', ridge)])
pipe.fit(X,y)
ridge.coef_
np.linalg.norm(ridge.coef_)

# Choose 2 lamda value，use validation method to estimate test error.
# X is not scale here in order to output the same results as the book. Normally have to scale. 
mse_test=[]
for alpha in (0.01, 1e10):
    validation=ShuffleSplit(random_state=0, test_size=0.5, n_splits=1)
    results=cross_validate(ElasticNet(alpha=alpha, l1_ratio=0),X,y,cv=validation, scoring='neg_mean_squared_error' )
    mse=-results['test_score']
    mse_test.append(mse)
mse_test

# GridsearchCV + validation
scaler=StandardScaler(with_mean=True, with_std=True)
ridge=ElasticNet(l1_ratio=0)
pipe=Pipeline(steps=[('scaler',scaler),('ridge', ridge)])
validation=ShuffleSplit(random_state=0, test_size=0.5, n_splits=1)
grid_val=GridSearchCV(pipe,param_grid={'ridge__alpha':lambdas},cv=validation, scoring='neg_mean_squared_error')
grid_val.fit(X, y)

# param_grid={'ridge__alpha':lambdas}  refer to: alpha of ridge. Since here we use pipeline as estimator.If the estimator is ridge, param_grid={'alpha':lambdas}.
grid_val.best_params_
grid_val.best_index_
# Output the best lambda 0.005887780525537749 and its index lambdas[75]
grid_val.best_estimator_
grid_val.best_score_
grid_val.cv_results_

# Use GridsearchCV + kfolds scoring =-mse
kfolds=KFold(random_state=0, n_splits=5, shuffle=True)
grid_kf=GridSearchCV(pipe,param_grid={'ridge__alpha':lambdas},cv=kfolds, scoring='neg_mean_squared_error')
grid_kf.fit(X, y)
grid_kf.best_estimator_
grid_kf.best_params_
# 0.011829922943770216
grid_kf.best_index_
grid_kf.best_score_
ax=sns.lineplot(x=-np.log(lambdas), y=(-grid_kf.cv_results_['mean_test_score']))
ax.set_xlabel('-log(lambdas)')
ax.set_ylabel('5folds test_mse')
# Better to add standard error in the plot. 

# Use GridsearchCV + kfolds scoring =R2  output lambda is the same in the method using -MSE as criteria
grid_r2=GridSearchCV(pipe, param_grid={'ridge__alpha':lambdas},cv=kfolds, scoring='r2')
grid_r2.fit(X, y)
grid_r2.best_estimator_
grid_r2.best_params_
#  0.011829922943770216
grid_r2.best_index_
grid_r2.best_score_
ax=sns.lineplot(x=-np.log(lambdas), y=(grid_r2.cv_results_['mean_test_score']))
ax.set_xlabel('-log(lambdas)')
ax.set_ylabel('5folds R2')
# Better to add standard error in the plot. 
# Output the best lambda 0.011829922943770216 and its index lambdas[72]

# Use ElasticNetCV + 5folds
ridgeCV=ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfolds)
pipeCV=Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
pipeCV.fit(X,y)
turned_ridge=pipeCV.named_steps['ridge']
mses=turned_ridge.mse_path_.mean(1)
best_mse=np.min(mses)
best_mse
turned_ridge.coef_
turned_ridge.alpha_

ax=sns.lineplot(x=-np.log(lambdas), y=mses)
ax.set_xlabel('-log(lambdas)')
ax.set_ylabel('5folds EN_CV_test_mse')
ax.axvline(-np.log(turned_ridge.alpha_), c='k', ls='--')

# Use ElasticNetCV + 5folds + validation
outer_val=ShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
inner_cv=KFold(n_splits=5, random_state=2, shuffle=True)
ridgeCV=ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=inner_cv)
pipeCV=Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
pipeCV.fit(X,y)

results=cross_validate(pipeCV, X, y,cv=outer_val, scoring= 'neg_mean_squared_error')
-results['test_score']
# 132384.13162359

# lasso
# Coefficient plots 
lasso_path=ElasticNet.path(X_scaled, y, l1_ratio=1, alphas=lambdas)
lasso_path[1].shape
coef_path_lasso=pd.DataFrame(lasso_path[1].T, columns=df.columns, index=-np.log(lambdas))
coef_path_lasso.index.name='-log(lambdas)'
coef_path_lasso.head()

for i in coef_path_lasso.columns:
    ax=sns.lineplot(x=coef_path_lasso.index, y=coef_path_lasso[i], label=i)
    ax.set_ylabel('Standardized lasso coefficients')
    ax.legend(fontsize=6,bbox_to_anchor= (1.2,1))

# ElasticNetCV and pipeCV, search the best lambda
lassoCV=ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=kfolds)
pipeCV=Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(X,y)
turned_lasso=pipeCV.named_steps['lasso']
turned_lasso.alpha_
# The best lambda 3.1421313804148743
turned_lasso.coef_

# Using best lambda and all data, 6 coefficients = 0 
mses=turned_lasso.mse_path_.mean(1)
best_mse=np.min(mses)
best_mse
# 114671.74998884085
ax=sns.lineplot(x=-np.log(lambdas), y=mses)
ax.set_xlabel('-log(lambdas)')
ax.set_ylabel('5folds EN_Lasso_CV_test_mse')
ax.axvline(-np.log(turned_lasso.alpha_), c='k', ls='--')

# Use ElasticNetCV lasso + 5folds+validation
outer_val=ShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
inner_cv=KFold(n_splits=5, random_state=2, shuffle=True)
lassoCV=ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=inner_cv)
pipeCV=Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(X,y)
results=cross_validate(pipeCV, X, y,cv=outer_val, scoring= 'neg_mean_squared_error')
-results['test_score']
# 133149.59033484

# V PCA
linreg=LinearRegression()
pca=PCA(n_components=2)
pipe=Pipeline(steps=[('scaler',scaler),('pca', pca),('linreg', linreg)])
pipe.fit(X,y)
pipe.named_steps['linreg'].coef_

# Gridsearch for components
grid_pca=GridSearchCV(pipe, param_grid={'pca__n_components':range(1,20)},cv=kfolds, scoring='neg_mean_squared_error')
grid_pca.fit(X, y)
grid_pca.best_estimator_
grid_pca.best_params_
# 17
grid_pca.best_score_
# 116222.02176741106
ax=sns.lineplot(x=np.arange(1,20), y=(-grid_pca.cv_results_['mean_test_score']))
ax.set_xlabel('# Principle components)')
ax.set_xticks(np.arange(1,20)[::1])
ax.set_ylabel('5folds test_mse')
ax.set_ylim([100000,250000])
# To have a simple clear comparison, set the ylim.  
# The best params is 17. But mse maintains constant, indicating 1 PCA is sufficient.

pipe.named_steps['pca'].explained_variance_ratio_
# The first 2 PCA explain 0.60155315 of variance
np.cumsum(pipe.named_steps['pca'].explained_variance_ratio_)
# There are only 2 components in pipe since it has been fitted with x, y. 
# Pipe in gridsearchCV has not been fitted with x and y.

# PLS
pls=PLSRegression(scale=True)
grid_pls=GridSearchCV(pls, param_grid={'n_components':range(1,20)},cv=kfolds, scoring='neg_mean_squared_error')
grid_pls.fit(X, y)
grid_pls.best_estimator_
grid_pls.best_params_
# 12 
grid_pls.best_score_
# 114684.61271597291
ax=sns.lineplot(x=np.arange(1,20), y=(-grid_pls.cv_results_['mean_test_score']))
ax.set_xlabel('# Principle components)')
ax.set_xticks(np.arange(1,20)[::1])
ax.set_ylabel('5folds test_mse')
ax.set_ylim([100000,250000])
# Param 12 is the lowest, but not much better than others

# Extra reading:
# Scikit-learn indeed does not support stepwise regression. That's because what is commonly known as 'stepwise regression' is an algorithm based on p-values of coefficients of linear regression, and scikit-learn deliberately avoids inferential approach to model learning (significance testing etc). Moreover, pure OLS is only one of numerous regression algorithms, and from the scikit-learn point of view it is neither very important, nor one of the best.

# There are, however, some pieces of advice for those who still need a good way for feature selection with linear models:
# Use inherently sparse models like ElasticNet or Lasso. Normalize your features with StandardScaler, and then order your features just by model.coef_. For perfectly independent covariates it is equivalent to sorting by p-values. The class sklearn.feature_selection.RFE will do it for you, and RFECV will even evaluate the optimal number of features. Use an implementation of forward selection by adjusted R2 that works with statsmodels.
# Do brute-force forward or backward selection to maximize your favorite metric on cross-validation (it could take approximately quadratic time in number of covariates). A scikit-learn compatible mlxtend package supports this approach for any estimator and any metric.
# If you still want vanilla stepwise regression, it is easier to base it on statsmodels, since this package calculates p-values for you. A basic forward-backward selection could look like this:""" 