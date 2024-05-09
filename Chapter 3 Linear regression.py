import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
sns.set_theme()

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm


from ISLP import load_data

Boston = load_data('Boston')
Boston.columns
Boston.head()
Boston.info()
# 506 by 13,non null, all numeric
Boston.describe()

# I Simple linear regression
x=Boston['lstat']
y=Boston['medv']
x=pd.DataFrame({'intercept' : np.ones(Boston.shape[0]), 'lstat' : Boston['lstat']})
# Put intercept b in coefficients y = wx+b  y = wx
# Using transform() get the same results

cor=x['lstat'].corr(y)
cor
sns.relplot(x=x['lstat'], y=y)
# See the possible linear relationship between x and y, and outliers

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=10)
x_train
type(x_train)

# Ordinary least squares OLS
model=sm.OLS(y_train, x_train).fit()
model.summary()
# Look at R square, coefficients, p-values, t-stat, std  error, F-stat
model.params
predictions=model.get_prediction(x_train)
#predictions contain various informations
predictions.predicted_mean[:3,]
predictions.conf_int(alpha=0.05)[:3,]
predictions.conf_int(alpha=0.05, obs=True)[:3,]
# confidence interval for mean and point estimation

x_train['lstat']
y_train

def abline (ax, b, m):
    xlim=ax.get_xlim()
    ylim=[m*xlim[0]+b, m*xlim[1]+b]
    ax.plot(xlim, ylim)
# Run several lines together, in one time  

ax=sns.scatterplot(x=x_train['lstat'], y=y_train)
abline(ax, model.params[0], model.params[1])

ax=sns.scatterplot(x=model.fittedvalues, y=model.resid)
ax.axhline(0, c='red', ls='--')
ax.set_xlabel('fitted value')
ax.set_ylabel('residual')
ax.set_title('Residual Plot')
# Residual plot, expecting evenly distributed on both sides of the X-axis 

inf1=model.get_influence().hat_matrix_diag
ax=sns.scatterplot(x=x_train.index, y=inf1)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
ax.set_title('Leverage Plot')

np.argmax(inf1)
# Return the index of the max-value

# we can use linearRegression to fit the model as well, but less convenient to output the statistics values
# change series to 2D array
x_train=x_train.values.reshape(-1,1)
x_train.shape  
y_train=y_train.values.reshape(-1,1)
model=LinearRegression().fit(x_train, y_train)
y_train_predict=model.predict(x_train)
model.intercept_
model.coef_


# II 2 features linear regression
x=Boston[['lstat','age']]
y=Boston['medv']
x=pd.DataFrame({'intercept' : np.ones(Boston.shape[0]), 'lstat' : Boston['lstat'], 'age': Boston['age']})

model1=sm.OLS(y, x).fit()
model1.params
model1.summary()

x=Boston[['lstat','age']]
y=Boston['medv']
x=pd.DataFrame({'intercept' : np.ones(Boston.shape[0]), 'lstat' : Boston['lstat'], 'age': Boston['age']})

# III Multi features linear regression
terms=Boston.drop('medv', axis=1)
terms['intercept']=np.ones(Boston.shape[0])
terms.head()
model2=sm.OLS(y, terms).fit()
model2.params
model2.summary()

# IV Exclude age and indus
terms1=Boston.drop(['medv','age','indus'],axis=1)
terms1['intercept']=np.ones(Boston.shape[0])
terms1.head()
model3=sm.OLS(y, terms1).fit()
model3.summary()

# V model goodness: R square, RSE, residual standard error
model3.scale
model3.rsquared    
np.sqrt(model3.scale)
# sqrt (square root)  
# RSE is a measure of lack of fit of the model to the data at hand. 

# VI Variance Inflation Factor，VIF: explain variance ratio bw having multicollinearity and no multicollinearity

terms1.head()
vals= [VIF(terms1, i) for i in range(terms1.shape[1]-1)]
# Exclude intercept columns
terms1.columns
vals
vif=pd.DataFrame(vals, columns= ['VIF'])
vif.set_index(terms1.columns[0:10], inplace=True)
vif

# VII ANNOVA
x=Boston[['lstat','age']]
y=Boston['medv']
x=pd.DataFrame({'intercept' : np.ones(Boston.shape[0]), 'lstat' : Boston['lstat'], 'age': Boston['age']})
x['lstat **2']=x['lstat']*x['lstat']
x.head()

model_a=sm.OLS(y, x.iloc[:, 0:2]).fit()
model_b=sm.OLS(y, x).fit()
model_a.summary()
model_b.summary()
anova_lm(model_a, model_b)

ax=sns.scatterplot(x=model_b.fittedvalues, y=model_b.resid)
ax.axhline(0, c='red', ls='--')
ax.set_xlabels("Fitted values")
ax.set_ylabels("Residual")

Carseats=load_data('Carseats')
Carseats.head()
Carseats.info()
y=Carseats.pop('Sales')

Carseats.ShelveLoc.unique()
Carseats['ShelveLoc']=pd.factorize(Carseats.ShelveLoc)[0].reshape(-1,1)
Carseats['intercept']=np.ones(Carseats.shape[0])
Carseats.Urban.unique()
Carseats.Urban=Carseats.Urban.map({"Yes":1,"No":1})
Carseats.us=Carseats.US.map({"Yes":1,"No":1})
Carseats.head()
Carseats.corr()

model=sm.OLS(y,Carseats).fit()
model.summary()

