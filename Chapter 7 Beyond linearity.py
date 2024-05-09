import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import (StandardScaler)
from statsmodels.api import (OLS, GLM, families)
from statsmodels.stats.anova import anova_lm
from statsmodels.gam.api import BSplines, GLMGam
from pygam import (s as s_gam, l as l_gam, f as f_gam, LinearGAM, LogisticGAM)

from ISLP import load_data
from ISLP.models import(bs, ns, poly, ModelSpec as MS)
from ISLP.transforms import (BSpline, NaturalSpline)
from ISLP.pygam import (approx_lam, degrees_of_freedom, plot as plot_gam, anova as anova_gam)

wage=load_data('Wage')
df=wage.copy()
df.head()
df.info()
df.describe()

sns.displot(df['wage'])
y=df.pop('wage')

# I Polynomial and anova test
# For raw parameter in poly function of ISL, if raw = true, produce basic polynomial features without scale; default=False,when raw=False, perform a QR decomposition on the resulting matrix of powers of centered and/or scaled features (scale+Orthogonal polynomials). 
poly_age=MS([poly('age', degree=4, raw=False)]).fit(df)
X=poly_age.transform(df)
X[:5]
model1=OLS(y, X).fit()
model1.summary()

age_grid = np.linspace(df['age'].min(),df['age'].max(),100)
age_to_pred= pd.DataFrame({'age': age_grid})
X_to_pred=poly_age.transform(age_to_pred)
X_to_pred.head()
preds=model1.get_prediction(X_to_pred)
bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]
preds.predicted_mean[:5]

sns.scatterplot(x=df['age'], y=y, alpha=0.5)
ax=sns.lineplot(x=age_grid, y=preds.predicted_mean)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.5,label='95% Ci')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')
ax.set_title('Degree_4 polynomial')

# Choose degree of polynomial or nested models with anova test
poly1=[MS([poly('age', degree=i)]) for i in range(1,6)]
X_poly1=[i.fit_transform(df) for i in poly1]
models=[OLS(y, X).fit() for X in X_poly1]
anova_lm(*models)
# Seconde example   can use cross val to select model as well
poly2=[MS(['education', poly('age', degree=i)]) for i in range(1,6)]
X_poly2=[i.fit_transform(df) for i in poly2]
models2=[OLS(y, X).fit() for X in X_poly2]
anova_lm(*models2)

# Binary classification, wether an individual earns more than 250000 dollars per year
df['high_earn']=y>250
df['high_earn'].head()
glm=GLM(df['high_earn'], X, family=families.Binomial()).fit()
glm.summary()
preds=glm.get_prediction(X_to_pred)
bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]

ax=sns.lineplot(x=age_grid, y=preds.predicted_mean)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.2,label='95% Ci')
rng=np.random.default_rng(0)
plt.scatter(df['age']+0.2*rng.uniform(size=y.shape[0]), np.where(df['high_earn']==True, 0.198, 0.002), marker='|')
ax.set_ylim([0,0.2])
ax.set_xlabel('Age')
ax.set_ylabel('P(wage>250)')
ax.set_title('Degree_4 polynomial')
# sns.rugplot(expand_margins=True) and displot(rug=true) probably works as well. 
# ylim([0,0.2]  in scatter plot x=age+random to avoid overlapping，y=(0.198-0.002)  
# There are only 79 high earns,so model has high variance and broad confidence interval (need more data)

# II Step function qcut quantile cut   pd.cut()
# For on variable, prediction is the mean of each interval. Intercept is no need. 
age_cut=pd.qcut(df['age'],4)
age_cat=pd.get_dummies(age_cut)
age_cat.head()
model_step=OLS(y,age_cat).fit()
model_step.summary()

age_grid = np.linspace(df['age'].min(),df['age'].max(),100)
age_to_pred= pd.qcut(age_grid, 4)
X_to_pred=pd.get_dummies(age_to_pred)
X_to_pred.head()
preds=model_step.get_prediction(X_to_pred)
preds.predicted_mean[:5]

bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]

fig,ax=plt.subplots(figsize=(4,4))
ax=sns.lineplot(x=age_grid, y=preds.predicted_mean, color='blue', linewidth=1)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.3,label='95% Ci',color='red')
sns.scatterplot(x=df['age'], y=y, alpha=0.5, fc='gray')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')
ax.set_title('Step functions of age')

# Classification high earn
df['high_earn']=y>250
df['high_earn'].head()
glm=GLM(df['high_earn'], age_cat, family=families.Binomial()).fit()
glm.summary()
preds=glm.get_prediction(X_to_pred)
bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]

fig,ax=plt.subplots(figsize=(4,4))
ax=sns.lineplot(x=age_grid, y=preds.predicted_mean)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.2,label='95% Ci')
rng=np.random.default_rng(0)
plt.scatter(df['age']+0.2*rng.uniform(size=y.shape[0]), np.where(df['high_earn']==True, 0.198, 0.002), marker='|')
ax.set_ylim([0,0.2])
ax.set_xlabel('Age')
ax.set_ylabel('P(wage>250)')
ax.set_title('Step functions of age')
# plots (output is different from book, dunno why)

# III Regression spline    Default is cubic, 3 nodes + cubic equation + intercept = 7 coefficients
# BSpline   While intercept=True, no need to add intercept in regression step. BSpline(df=6), we can set Degree of freedom
bs_=BSpline(internal_knots=[25,40,60], intercept=True).fit(df['age'])
bs_age=bs_.transform(df['age'])
bs_age.head()
bs_age
model_sp=OLS(y, bs_age).fit()
model_sp.summary()

# bs can create intercept by default. But we need intercept for regression, so add intercept additionally 
bs_age = MS([bs('age', internal_knots=[25,40,60])])
Xbs = bs_age.fit_transform(df)
Xbs.head()
M = OLS(y, Xbs).fit()
M.summary()

# 3 nodes, 0th power  = step qcut
bs_age = MS([bs('age',df=3, degree=0)]).fit(df)
Xbs = bs_age.transform(df)
model=OLS(y, Xbs).fit()
model.summary()

# natural spline  
# df=5 i.e 4 nodes in line, 2 nodes for 2 ends. In total, 6 nodes. 6+3-2*2=5   df of model = 5+intercept=6
bs_age = MS([ns('age',df=5)]).fit(df)
Xbs = bs_age.transform(df)
model=OLS(y, Xbs).fit()
model.summary()

age_grid = np.linspace(df['age'].min(),df['age'].max(),100)
age_to_pred= pd.DataFrame({'age': age_grid})
X_to_pred=bs_age.transform(age_to_pred)
X_to_pred.head()
preds=model.get_prediction(X_to_pred)
preds.predicted_mean[:5]

bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]

fig,ax=plt.subplots(figsize=(4,4))
ax=sns.lineplot(x=age_grid, y=preds.predicted_mean, color='blue', linewidth=0.25)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.3,label='95% Ci', color='red')
sns.scatterplot(x=df['age'], y=y, alpha=0.5, fc='gray')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')
ax.set_title('Cubic natural spline with 6 knots (including 2 boundary knots)')

# High earn classification 
df['high_earn']=y>250
df['high_earn'].head()
glm=GLM(df['high_earn'], Xbs, family=families.Binomial()).fit()
glm.summary()
preds=glm.get_prediction(X_to_pred)
bands=preds.conf_int(alpha=0.05)
lower_ci=bands[:,0]
upper_ci=bands[:,1]

fig,ax=plt.subplots(figsize=(4,4))
ax=sns.lineplot(x=age_grid, y=preds.predicted_mean)
ax.fill_between(age_grid, upper_ci,lower_ci,alpha=0.2,label='95% Ci')
rng=np.random.default_rng(0)
plt.scatter(df['age']+0.2*rng.uniform(size=y.shape[0]), np.where(df['high_earn']==True, 0.198, 0.002), marker='|')
ax.set_ylim([0,0.2])
ax.set_xlabel('Age')
ax.set_ylabel('P(wage>250)')
ax.set_title('Cubic natural spline with 6 knots (including 2 boundary knots)')

# IV Smoothing splines and GAMS (generalized additive models)
# If lambda is small, rough lines and many bends; while lambda is large, spline is smooth, close to straight line
# Gam accept 2D array, so transform X to 2D
X_age=np.asarray(df['age']).reshape(-1,1)
X_age[:5]
y[:5]

for lam in np.logspace(-2, 6, 5):
    gam=LinearGAM(s_gam(0, lam=lam))
    model=gam.fit(X_age, y)
    pred=model.predict(age_grid)
    ax=sns.lineplot(x=age_grid, y=pred, label=f'lambda={lam:.1e}')
sns.scatterplot(x=df['age'], y=y, alpha=0.5, fc='gray')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')

gam_opt=gam.gridsearch(X_age, y)
sns.lineplot(x=age_grid, y=gam_opt.predict(age_grid), label='grid search')
# Grid search for lambda based on x and y

# approx_lam and degrees_of_freedom are 2 functions of ISLP
# The parameter in approx 4 is the df of model, including the df 1 of intercept
# Fit the data in the model without defining lambda value, then set df=4, output the model lambda value via approx_lam function
gam2=LinearGAM(s_gam(0))
gam2.fit(X_age, y)
age_term=gam2.terms[0]
age_term.lam=approx_lam(X_age, age_term, 4)
age_term.lam
# In contrast, with lambda value known，we can compute df value via degrees_of_freedom function
degrees_of_freedom(X_age, age_term)

# Fit firstly, then set lambda value, fit again
gam1=LinearGAM(s_gam(0))
gam1.fit(X_age, y)    
age_term=gam1.terms[0]
for degree in [1,3,4,8,15]:  
    age_term.lam=approx_lam(X_age, age_term, degree+1)
    gam1.fit(X_age, y)
    print(degrees_of_freedom(X_age, age_term))
    ax=sns.lineplot(x=age_grid, y=gam1.predict(age_grid), label=f'df{degree}')
sns.scatterplot(x=df['age'], y=y, alpha=0.5, fc='gray')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')
ax.set_title('Degree of freedom')

# V Additive models with several terms using natural spline

# NS  NaturalSpline  default intercept = Fault  
# Visualize relationship bw 3 features and y  
# Education is categorical, use step function， year and age are continuous variables, use NS
sns.scatterplot(x=df['year'], y=y, alpha=0.5, fc='gray')
sns.scatterplot(x=df['education'], y=y, alpha=0.5, fc='gray')
sns.scatterplot(x=df['age'], y=y, alpha=0.5, fc='gray')

ns_age=NaturalSpline(df=4, intercept=False).fit(df['age']) 
X_age=ns_age.transform(df['age'])
X_age.head()
X_age.shape
X_age.columns=['age1','age2','age3','age4']
X_age.head()
# Rename columns for better distinguish after transformation, and not delete several cols after concate. NS default setting, intercept =False
ns_year=NaturalSpline(df=5).fit(wage['year']) 
X_year=ns_year.transform(wage['year'])
X_year.head()
X_year.shape
X_year.columns=['year0','year1','year2','year3','year4']
X_year.head()
# df4 for year? dunno why 5

dum_edu=pd.get_dummies(df['education'])
dum_edu.head()
dum_edu.shape
# dum_edu take attention on coefficient explanation, for get_dummies function, default setting first_drop=False.  
X_ns=pd.concat([X_age, X_year, dum_edu], axis=1)
X_ns.head()

gam_manual=OLS(y,X_ns).fit()
gam_manual.summary()
# Why intercept is not added here？

# Influence of age on wage  partial dependence 
# Prepare 100 age values  set year and education values as their average. 
# The 4 columns of age= age_grid
X_means=X_ns.mean(axis=0).values
X_means
X_age_ns= X_ns[:100]
X_age_ns[:]=X_means
X_age_ns.head()
X_age_ns.drop(X_age_ns.columns[0:4],axis=1, inplace=True)
X_age_ns.head()
age_grid = np.linspace(df['age'].min(),df['age'].max(),100)
age_to_pred=pd.DataFrame(ns_age.transform(age_grid), columns=X_age.columns)
age_to_pred.head()
X_age_ns=pd.concat([age_to_pred,X_age_ns], axis=1)
X_age_ns.head()
X_age_ns.shape

age_preds=gam_manual.get_prediction(X_age_ns)
bands_age=age_preds.conf_int(alpha=0.05)
partial_age=age_preds.predicted_mean
center=partial_age.mean()
partial_age=partial_age-center
bands_age=bands_age-center
# - average of prediction 

ax=sns.lineplot(x=age_grid,y=partial_age)
sns.lineplot(x=age_grid,y=bands_age[:,0],ls='--')
sns.lineplot(x=age_grid,y=bands_age[:,1],ls='--')
ax.set_xlabel('Age')
ax.set_ylabel('Effect on age')
ax.set_title('Partial dependence of age on wage')

# Influence of year on wage. 
# Take attention, we have to replace the 5 columns of year at their original positions, in order to be consistent with the model.
X_means=X_ns.mean(axis=0).values
X_means
X_year_ns= X_ns[:100]
X_year_ns[:]=X_means
X_year_ns.head()
X_year_ns.columns
year_features=['year0', 'year1', 'year2', 'year3','year4']
year_grid = np.linspace(df['year'].min(),df['year'].max(),100)
X_year_ns[year_features]=ns_year.transform(year_grid)
X_year_ns.head()
X_year_ns.shape

year_preds=gam_manual.get_prediction(X_year_ns)
bands_year=year_preds.conf_int(alpha=0.05)
partial_year=year_preds.predicted_mean
center=partial_year.mean()
center
partial_year=partial_year-center
bands_year=bands_year-center

ax=sns.lineplot(x=year_grid,y=partial_year)
sns.lineplot(x=year_grid,y=bands_year[:,0],ls='--')
sns.lineplot(x=year_grid,y=bands_year[:,1],ls='--')
ax.set_xlabel('Year')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of year on wage')

# Using smoothing spline 
df['education'].info()
df['education'].cat.codes
# Encode cat features
# Apply spline to age and year. 7 spines for 7 years. 
# Education: categorical spline, step-function, linear, lam=0, slope is consistent. 
# Default lam =0.6，
# (0 1 2): columns index.
# np.column_stack: combine several series into a array
gam_full=LinearGAM(s_gam(0)+s_gam(1, n_splines=7)+f_gam(2, lam=0))
X_gam=np.column_stack([df['age'], df['year'], df['education'].cat.codes])
X_gam[:5]
gam_full=gam_full.fit(X_gam, y)
gam_full.summary()

# plot_gam() is a ISLP function. It needs 2 parameters: estimator and feature index. 
# We see that curve is not smooth using default lam
fig,ax=plt.subplots(figsize=(8,8))
plot_gam(gam_full, 0, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage- default lam=0.6')

# Did not use plot function of ISLP here 
age_grid=gam_full.generate_X_grid(term=0)
age_grid
pdep, confi=gam_full.partial_dependence(term=0, X=age_grid, width=0.95)

ax=sns.lineplot(x=age_grid[:,0], y=pdep)
sns.lineplot(x=age_grid[:,0], y=confi[:,0],ls='--')
sns.lineplot(x=age_grid[:,0], y=confi[:,1],ls='--')
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage- default lam=0.6')

# Define degree of freedom, compute lam via ISLP function, feed it in GAM, refit, then plot. Curve is smoother.
age_term=gam_full.terms[0]
age_term.lam=approx_lam(X_gam, age_term, df=4+1)
year_term=gam_full.terms[1]
year_term.lam=approx_lam(X_gam, year_term, df=4+1)
gam_full.fit(X_gam,y)

age_grid=gam_full.generate_X_grid(term=0)
age_grid
pdep, confi=gam_full.partial_dependence(term=0, X=age_grid, width=0.95)

ax=sns.lineplot(x=age_grid[:,0], y=pdep)
sns.lineplot(x=age_grid[:,0], y=confi[:,0],ls='--')
sns.lineplot(x=age_grid[:,0], y=confi[:,1],ls='--')
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage- df=5 including intercept')

# Plot year partial dependence using ISLP plot function
fig,ax=plt.subplots(figsize=(8,8))
plot_gam(gam_full, 1, ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of Year on wage- df=5 including intercept')

# Plot education partial dependence using ISLP plot function
fig, ax = plt.subplots()
ax = plot_gam(gam_full, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of wage on education',fontsize=20);
ax.set_xticklabels(df['education'].cat.categories, fontsize=8)

# Plot
for i, j in enumerate(gam_full.terms):
    if j.isintercept:
        continue
    XX = gam_full.generate_X_grid(term=i)
    pdep, confi = gam_full.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, j.feature], pdep)
    plt.plot(XX[:, j.feature], confi, c='r', ls='--')
    plt.title(f'Partial dependence of {repr(j)} on wage')
    plt.xlabel(repr(j))
    plt.ylabel('Effect on wage')
    plt.show()

# Anova test for GAM models: for year term, linear model is enough
year_0=LinearGAM(age_term+f_gam(2, lam=0)) 
year_0.fit(X_gam, y)
year_0.terms
year_0.fit(X_gam,y)

year_linear=LinearGAM(age_term+l_gam(1, lam=0)+f_gam(2, lam=0)) 
year_linear.fit(X_gam, y)
year_linear.terms

anova_gam(year_0, year_linear, gam_full)

# For age term, spline is necessary
age_0=LinearGAM(year_term+f_gam(2, lam=0)) 
age_0.fit(X_gam, y)
age_0.terms

age_linear=LinearGAM(l_gam(0, lam=0)+year_term+f_gam(2, lam=0)) 
age_linear.fit(X_gam, y)
age_linear.terms

anova_gam(age_0, age_linear, gam_full)

# Compare params of s(age)+l(year)+f(education) and full_gam. Both of R2 and AIC are comparable, choose simple model
year_linear.summary()
gam_full.summary()
  
# Make prediction with training data
pred_train=year_linear.predict(X_gam)
pred_train.shape

# Logistic regression GAM
high_earn=LogisticGAM(age_term+l_gam(1, lam=0)+f_gam(2, lam=0))
high_earn=high_earn.fit(X_gam, df['high_earn'].values)

# Plot in one time
for i, j in enumerate(high_earn.terms):
    if j.isintercept:
        continue
    XX = high_earn.generate_X_grid(term=i)
    pdep, confi = high_earn.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, j.feature], pdep)
    plt.plot(XX[:, j.feature], confi, c='r', ls='--')
    plt.title(f'Partial dependence of high wage on {repr(j)}')
    plt.xlabel(repr(j))
    plt.ylabel('Effect on wage')
    plt.show()
# Look at high earn and education data, no observation for the first class, so ci is large  
# 268 observations in this class
pd.crosstab(df['high_earn'], df['education'])

# Exclude this class and retrain with linear year
zero_high=(df['education']=='1. < HS Grad')
zero_high
df2=df.loc[~zero_high]
df2.shape

X_high=np.column_stack([df2['age'], df2['year'], df2['education'].cat.codes-1])

high_earn=LogisticGAM(age_term+l_gam(1, lam=0)+f_gam(2, lam=0))
high_earn=high_earn.fit(X_high, df2['high_earn'].values)

fig, ax = plt.subplots()
ax = plot_gam(high_earn, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of wage on education');
ax.set_xticklabels(df2['education'].cat.categories[1:], fontsize=8);

# Exclude this class and retrain with spline year
high_earn=LogisticGAM(age_term+year_term+f_gam(2, lam=0))
high_earn=high_earn.fit(X_high, df2['high_earn'].values)

fig, ax = plt.subplots()
ax = plot_gam(high_earn, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of wage on education');
ax.set_xticklabels(df2['education'].cat.categories[1:], fontsize=8);

for i, j in enumerate(high_earn.terms):
    if j.isintercept:
        continue
    XX = high_earn.generate_X_grid(term=i)
    pdep, confi = high_earn.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, j.feature], pdep)
    plt.plot(XX[:, j.feature], confi, c='r', ls='--')
    plt.title(f'Partial dependence of high wage on {repr(j)}')
    plt.xlabel(repr(j))
    plt.ylabel('Effect on wage')
    plt.show()


# poly_age=np.power.outer(np.array(df['age']), np.arange(5))
# poly_age=pd.DataFrame(poly_age, columns=['intercept','degree1','degree2','degree3','degree4']
# scaler=StandardScaler()
# poly_age.iloc[:,1:5]=scaler.fit_transform(poly_age.iloc[:,1:5])
# poly_age.head()
# scaler.fit_transform(poly_age['degree1','degree2','degree3','degree4'])
# poly_age.head()
# model1=OLS(y, poly_age1).fit()
# model1.summary()


