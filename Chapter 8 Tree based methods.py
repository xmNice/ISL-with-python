import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

from statsmodels.datasets import get_rdataset

from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree, export_text)
from sklearn.ensemble import (RandomForestRegressor as RF, GradientBoostingRegressor as GBR)
from sklearn.model_selection import (ShuffleSplit, train_test_split, KFold, cross_validate, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, classification_report)
from sklearn.

from ISLP import (load_data, confusion_table)
from ISLP.models import ModelSpec as MS
from ISLP.bart import BART

carseats=load_data('Carseats')
df=carseats.copy()
df.head()
df.info()
df.describe()
y=df.pop('Sales')
X=df.copy()
X=pd.get_dummies(X, drop_first=True)
X.head()
feature_names=list(X.columns)
feature_names

# I Classification trees using decisiontree classifier
y_high= np.where(y>8, 'Yes', 'No')
y_high[:5]

# model = MS(df.columns, intercept=False
# D = model.fit_transform(df)

clf_d3=DTC(criterion='entropy', max_depth=3, random_state=0).fit(X, y_high)

clf_d3.tree_.n_leaves
importance_d3=clf_d3.feature_importances_
indices = np.argsort(importance_d3)[::-1]

# Extract index after sorting, default in ascending 
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (indices[f], 30, feature_names[indices[f]], importance_d3[indices[f]]))   
ax=sns.barplot(x=np.arange(len(feature_names)), y=importance_d3[indices], color='lightblue')
ax.set_xticks(range(X.shape[1]), X[feature_names], rotation=90)
plt.title('Feature Importance')

accuracy_score(y_high, clf_d3.predict(X))
# 0.79  training error rate = 1-0.79 = 0.21
residual_devirance=np.sum(log_loss(y_high, clf_d3.predict_proba(X)))
residual_devirance
# 0.4710647062649358 
# Similar to entropy and ginindex, model evaluation metric, the lower the better

# Plot tree 
ax=plt.subplot()
plot_tree(clf_d3, feature_names=list(X.columns),  fontsize=6)
print(export_text(clf_d3, feature_names=list(X.columns), show_weights=True))

# Cross val  
validation=ShuffleSplit(random_state=0, n_splits=1, test_size=200)
results=cross_validate(clf_d3, X, y_high, cv=validation)
results['test_score']
#  0.685  test score is accuracy? If yes, this score is lower than training score, this tree is overfitting , need to be pruned. 

# Retrain, use half of data to train a full tree and inspect test error
X_train, X_test, y_train, y_test=train_test_split(X, y_high, random_state=0, test_size=0.5)
X_train.head()
clf_full=DTC(random_state=0, criterion='entropy').fit(X_train, y_train)
# Check full tree and leaves
print(export_text(clf_full, feature_names=list(X.columns), show_weights=True))
clf_full.tree_.n_leaves 
importance_full=clf_full.feature_importances_
# plot feature importance    We see that it  is different from that of D3 model why？
indices = np.argsort(importance_full)[::-1]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (indices[f], 30, feature_names[indices[f]], importance_full[indices[f]]))   
ax=sns.barplot(x=np.arange(len(feature_names)), y=importance_full[indices], color='lightblue')
ax.set_xticks(range(X.shape[1]), X[feature_names], rotation=90)
plt.title('Feature Importance')

# Check results given by full tree 35 leaves and 35 splits  agree with book 
print(classification_report(y_test, clf_full.predict(X_test)))
accuracy_score(y_test,clf_full.predict(X_test))
# 0.735
# Inverse 2 parameters gives the same results 

# Find out the best subtree from full tree
ccp_path=clf_full.cost_complexity_pruning_path(X_train, y_train)
kfold=KFold(n_splits=10, shuffle=True, random_state=1)
grid=GridSearchCV(estimator=clf_full, param_grid={'ccp_alpha':ccp_path.ccp_alphas},refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)
best_tree=grid.best_estimator_
grid.best_score_
# 0.685
importances_best=best_tree.feature_importances_
# Summarize feature importance
for i,v in enumerate(importances_best):
    print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance
sns.barplot(x= np.arange(len(importances_best)), y=importances_best)

plot_tree(best_tree, feature_names=list(X.columns),  fontsize=6)
print(export_text(best_tree, feature_names=list(X.columns), show_weights=True))
best_tree.tree_.n_leaves
#30
accuracy_score(y_test, best_tree.predict(X_test))
#0.72
confusion_table(best_tree.predict(X_test), y_test)
confusion_matrix(y_test, best_tree.predict(X_test))
# Confusion matrix of ISLP and sklearn: 1st parameter is x index, 2nd is y index. Report is more clear.
print(classification_report(y_test, best_tree.predict(X_test)))
# Compared to full tree, accuracy is increased from 0,68 to 0,72. cut 5 leaves

# Comparison of feature importance   full tree and best tree are nested. Cut off one feature gives the same pattern of feature importance. But for depth3 tree, feature importance is very different, especially feature 0 CompPrice has 0 importance, wired.
for i,v in enumerate(importance_d3):
    print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance
sns.barplot(x= np.arange(len(importance_d3)), y=importance_d3)

for i,v in enumerate(importance_full):
    print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance
sns.barplot(x= np.arange(len(importance_full)), y=importance_full)

for i,v in enumerate(importances_best):
    print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance
sns.barplot(x= np.arange(len(importances_best)), y=importances_best)

# II Fitting regression tree
boston=load_data('Boston')
df1=boston.copy()
df1.head()
df1.info()
df1.describe()

y=df1.pop('medv')
X=df1.copy()
X.head()
feature_names=list(X.columns)
feature_names

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)
# Train a full tree
reg=DTR().fit(X_train,y_train)
reg.tree_.n_leaves  
#337
importances_full=reg.feature_importances_
for i,v in enumerate(importances_full):
    print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance  'rm'seems the most important  followed by 'lstat'  features 1 3 8 are almost 0 importance
sns.barplot(x=np.arange(len(importances_full)), y=importances_full)

# Test error produced by full tree
test_pred_full= reg.predict(X_test)
mse_test_full=np.mean((y_test-test_pred_full)**2)
mse_test_full
# 27.70598684210526
ax=plt.subplot()
plot_tree(reg, feature_names=feature_names, ax=ax)


# Select the best subtree
ccp_path_reg=reg.cost_complexity_pruning_path(X_train, y_train)
kfold=KFold(n_splits=5, random_state=10, shuffle=True)
grid=GridSearchCV(estimator=reg, param_grid={'ccp_alpha':ccp_path_reg.ccp_alphas}, refit=True, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

best_tree_reg=grid.best_estimator_
# ccp_alpha=0.04249058380413948
best_tree_reg.tree_.n_leaves
# 66 leaves
importances_best=best_tree_reg.feature_importances_
for i,v in enumerate(importances_best):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance  'rm'seems the most important  followed by 'lstat'
sns.barplot(x=np.arange(len(importances_best)), y=importances_best)
# Importance is almost the same as full tree

# Test error produced by best subtree
test_pred_best= best_tree_reg.predict(X_test)
mse_test_best=np.mean((y_test-test_pred_best)**2)
mse_test_best
# 28.56971557009449

ax=plt.subplot()
plot_tree(best_tree_reg, feature_names=feature_names, ax=ax)
plt.savefig('best_tree', dpi=1000)
# Output tree agrees with book

# III Bagging and random forest
bag_boston=RF(max_features=len(feature_names), random_state=0, n_estimators=100)
bag_boston.fit(X_train,y_train)
test_pred_bag= bag_boston.predict(X_test)
mse_test_bag=np.mean((y_test-test_pred_bag)**2)
mse_test_bag
# 14.634700151315787 agree with book
sns.scatterplot(y=y_test, x=test_pred_bag)

# n_estimators=100 is default. If we set 500:
bag_boston500=RF(max_features=len(feature_names), random_state=0, n_estimators=500)
bag_boston500.fit(X_train,y_train)
test_pred_bag500= bag_boston500.predict(X_test)
mse_test_bag500=np.mean((y_test-test_pred_bag500)**2)
mse_test_bag500
# 14.605662565263161  no improvement, so 100 is enough

# Use random forest
rf_boston=RF(max_features=6, random_state=0, n_estimators=100)
rf_boston.fit(X_train,y_train)
test_pred_rf= rf_boston.predict(X_test)
mse_test_rf=np.mean((y_test-test_pred_rf)**2)
mse_test_rf
# 20.04276446710527 agree with book   little bit worse than bag
sns.scatterplot(y=y_test, x=test_pred_rf)

feature_imp=pd.DataFrame({ 'importance':rf_boston.feature_importances_}, index=feature_names)
feature_imp
feature_imp= feature_imp.sort_values(by='importance', ascending=False)
# lstat and rm are the most significant features
ax=sns.barplot(x=feature_imp['importance'], y=feature_imp.index)
ax.set_xlabel('Importance')
ax.set_ylabel('Feature names')
ax.set_title('Total decrease in node mse, averaged over all trees')

# GBR 
boost_boston=GBR(n_estimators=5000,learning_rate=0.001, max_depth=3, n_estimators=5000, learning_rate=0.001, max_depth=3,random_state=0)
boost_boston.fit(X_train, y_train)

x=np.arange(boost_boston.train_score_.shape[0])
train_errors=boost_boston.train_score_
test_errors=[]
for i in boost_boston.staged_predict(X_test):
    test_error=np.mean((y_test-i)**2)
    test_errors.append(test_error)
sns.lineplot(x=x, y=train_errors,label='Training error')
sns.lineplot(x=x, y=test_errors, label='Test error')
plt.ylabel('MSE')
plt.xlabel('Tree')
plt.title('Gradient Boosting Regression')

test_error=np.mean((y_test-boost_boston.predict(X_test))**2)
test_error
# 14.481405918831591

# In decision trees, feature importance is determined by how much each feature contributes to reducing the uncertainty in the target variable. This is typically measured by the amount of reduction in the Gini impurity or entropy that is achieved by splitting on a particular feature.

# for i,v in enumerate(importance_d3):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance 分成8个区
# sns.barplot(x= np.arange(len(importance_d3)), y=importance_d3)