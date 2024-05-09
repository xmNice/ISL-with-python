import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error)

from ISLP import load_data 

# I Read and understand the data
bikeshare=load_data('Bikeshare')
df=bikeshare.copy()
df.head()
df.info()
df.describe() 
# Observations: 8645 rows and 15 columns; 
# 3 categorical columns; all other are either float or integer type; 
# Some fields that are categorical in nature, but in integer/float type. 
# We will analyse and finalize whether to convert them to categorical or treat as integer;
# no NA values. OUTLIERS?

# II Check data quality and clean data (NA. duplicate, outliers)
# Percentage of missing values in each column
round(100*(df.isnull().sum()/len(df)),2) .sort_values(ascending=False)
# Row-wise null count percentage
round((df.isnull().sum(axis=1)/len(df))*100,2).sort_values(ascending=False)
# No NA values.

# Duplicate Check  no duplicate
df=df.drop_duplicates()
df.shape

# Dropping columns not required for modeling 
# ‘bikers’ variable indicates the total number of bike rentals, including both casual and registered, is the target variable.” 
# we have already month and weekday features

# III EDA
# III.1 Check MI
xx = bikeshare.copy()
yy = xx.pop("bikers")
# Label encoding for categories
for colname in xx.select_dtypes("category"):
    xx[colname], _ = xx[colname].factorize()
# All discrete features should now have integer dtype (double-check this before using MI!)
discrete_features = xx.dtypes == int
def make_mi_scores(xx, yy, discrete_features):
    mi_scores = mutual_info_regression(xx, yy, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=xx.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_scores = make_mi_scores(xx, yy, discrete_features)
mi_scores[::3]  # Show a few features with their MI scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

# Observations: most of users are registered, 3 times more than casual. 
# The 3 least important  features are 'windspeed', 'holiday', 'weathersit'.

# III.2 Explore numeric features 
# Corr bw numeric features: 'temp' and atemp have high linear correlation, can drop one of them. 'windspeed' seems to have no effect with target, leave it aside and see. The other features have intermediate linear relationship with the target.

num_features=['temp','atemp','hum','windspeed','bikers']

sns.boxplot(df['temp'])
sns.boxplot(df['atemp'])
sns.boxplot(df['hum'])
sns.boxplot(df['windspeed'])
sns.boxplot(df['bikers'])

corr=df[num_features].corr()
corr
# Visualize correlation by pairplot and heatmap 
sns.pairplot(df[num_features])
# Find some outliers 
sns.heatmap(corr, annot=True,cmap='Blues')

df.drop(['casual', 'registered','atemp'], inplace=True, axis=1)
# Drop some redundant features

# III.3 Explore categorical features
df.columns
cat_features=['season', 'mnth', 'hr','holiday', 'weekday', 'workingday', 'weathersit']
for i in cat_features:
    print(df[i].value_counts(), '\n')
# 'holidays' is a imbalanced feature

# Visualize their distribution
sns.boxplot(x=df['season'], y=df['bikers'])
sns.boxplot(x=df['mnth'], y=df['bikers'])
sns.boxplot(x=df['hr'], y=df['bikers'])
sns.boxplot(x=df['holiday'], y=df['bikers'])
sns.boxplot(x=df['weekday'], y=df['bikers'])
sns.boxplot(x=df['workingday'], y=df['bikers'])
sns.boxplot(x=df['weathersit'], y=df['bikers'])

# If we have 'year' data, we can explore yearly trend and seasonality.
# Drop 'day' feature to avoid too many dummies, moreover, daily information is represented by month and week. 
# Seasonal: Fall has the highest average rentals, followed closely by summer.
# Monthly trend: september tops the monthly rental count, with surrounding months showing substantial demand.The trend aligns with seasonal patterns, indicating a correlation between rentals and seasons.
# Holiday vs. working Days: holidays generally result in lower rental counts compared to working days.Holidays exhibit greater variability in rental demand.
# Weekday : Overall, no significant difference in rentals across weekdays is observed.Thursdays and Sundays stand out with higher variability in rental counts compared to other weekdays.

# Encode categorical features 'hr', 'weathersit', 'mnth'.
df.drop (['day'],inplace=True, axis=1)
df['season']=df['season'].astype('category')
df=pd.get_dummies(df,drop_first=True)
df.columns

# IV Train models
# Train with all features, select features based on p-values
df_train, df_test = train_test_split(df, test_size=0.3, random_state=10)
y_train=df_train.pop('bikers')
y_test=df_test.pop('bikers')

X_train.columns
scaler=StandardScaler()
to_scaler=['temp','hum','windspeed']

X_train=df_train
X_train[to_scaler]=scaler.fit_transform(X_train[to_scaler])
X_train.head()

X_test=df_test
X_test[to_scaler]=scaler.transform(X_test[to_scaler])

# IV.1 linear model 
X_train_lm=sm.add_constant(X_train)
X_train_lm.head()
model1=sm.OLS(y_train,X_train_lm).fit()
model1.summary()
# R2=0.68, remove 'weathersit_heavy rain/snow' (high p values and only one point in this class). Drop 'weath', 'holiday', 'windspead' as well. Then, retrain.

X_train_lm.drop(['weathersit_heavy rain/snow','holiday', 'windspeed', 'weathersit_cloudy/misty'],axis=1, inplace=True)
X_train_lm.columns
X_train_lm.shape
model2=sm.OLS(y_train,X_train_lm).fit()
model2.summary() 
# R2 is not changed.
X_train_lm.drop(['weekday','workingday'],axis=1, inplace=True)
X_train_lm.columns
model3=sm.OLS(y_train,X_train_lm).fit()
model3.summary() 
# R2 is not improved, P-values are all significant.

# Output coefficients for 'month', 'hours' and 'seasons', then drop some sub-features.
coef=model3.params
coef
coef[6:17]
ax=sns.lineplot(x=np.arange(2,13),y=coef[6:17].values,markers=True)
ax.set_xlabel('Month')
ax.set_ylabel('Coefficient')
# Max: may and september; min:january, july, feb. 

coef[17:40]
ax=sns.lineplot(x=np.arange(1,24),y=coef[17:40].values,markers=True)
ax.set_xlabel('Hour')
ax.set_ylabel('Coefficient')
# Max: hour 8 9 16 17 ; intermediate: 10-15, lowest: midnight. 

coef[3:6]
ax=sns.lineplot(x=np.arange(1,4),y=coef[3:6].values,markers=True)
ax.set_xlabel('Season')
ax.set_ylabel('Coefficient')
# winter lowest. 

# Drop some sub-features and train last model
X_train_lm.drop(['mnth_Feb','mnth_March', 'hr_1','mnth_July', 'mnth_Nov','mnth_Dec', 'mnth_Aug'],axis=1, inplace=True)
X_train_lm.columns
model4=sm.OLS(y_train,X_train_lm).fit()
model4.summary() 

# IV.2 Poisson regression
model_pois=sm.GLM(y_train,X_train_lm, families=sm.families.Poisson()).fit()
model_pois.summary()
coef_pois=model_pois.params
coef

fig,ax=plt.subplots()
x=model2.fittedvalues
y=model_pois.fittedvalues
x.shape
y.shape
ax.scatter(x=x, y=y, alpha=0.5)
ax.axline([0,0], c='red', slope=1)
# High positive correlation, nearly the same estimate

# V Evaluate models
# Linear model
# Residual analysis for train_set
y_train_pred= model4.predict(X_train_lm)
ax=sns.displot((y_train - y_train_pred), bins = 20, kde=True)
ax.set_xlabels('Error')

R2_train=r2_score(y_train,y_train_pred)
R2_train
MSE_train = np.sqrt(mean_squared_error(y_train,y_train_pred))
MSE_train

# Predictions for test set
X_test=sm.add_constant(X_test)
X_train_lm.columns
feature_keep=X_train_lm.columns
X_test[feature_keep].head()
linear_pred=model4.predict(X_test[feature_keep])

R2_test=r2_score(y_test,linear_pred)
R2_test
MSE_test = np.sqrt(mean_squared_error(y_test, linear_pred))
MSE_test

# R2 and MSE are the same for train and test set
ax=sns.relplot(x=y_test, y=linear_pred)
ax.set_xlabels('y_test')
ax.set_ylabels('linear_pred')

# poisson model
pois_pred=model_pois.predict(X_test[feature_keep])
ax=sns.relplot(x=y_test, y=pois_pred)
ax.set_xlabels('y_test')
ax.set_ylabels('pois_pred')