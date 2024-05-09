import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np
import pandas as pd

from lifelines import (KaplanMeierFitter, CoxPHFitter)
from lifelines.statistics import (logrank_test, multivariate_logrank_test)

from ISLP.survival import sim_time
from ISLP import load_data


Working on it....

BrainCancer = load_data('BrainCancer')
BrainCancer.columns
BrainCancer.head()

BrainCancer['sex'].value_counts()
BrainCancer['diagnosis'].value_counts()
BrainCancer['status'].value_counts()

km = KaplanMeierFitter()
km_brain = km.fit(BrainCancer['time'], BrainCancer['status'])
fig, ax = plt.subplots(figsize=(8,8))
km_brain.plot(label='Kaplan Meier estimate', ax=ax)

fig, ax = plt.subplots(figsize=(8,8))
for sex, df in BrainCancer.groupby('sex'):
    km_sex = km.fit(df['time'], df['status'])
    km_sex.plot(label='Sex=%s' % sex, ax=ax)

 
logrank_test(by_sex['Male']['time'],by_sex['Female']['time'],by_sex['Male']['status'],by_sex['Female']['status'])