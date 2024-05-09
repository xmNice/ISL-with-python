import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import (KMeans, AgglomerativeClustering)
from scipy.cluster.hierarchy import (dendrogram, cut_tree, linkage) # fonction exists in sklean?

from statsmodels.datasets import get_rdataset
from ISLP import load_data
#from ISLP.cluster import compute_linkage

# KMeans
np.random.seed(0)
X=np.random.standard_normal ((50,2))
X.shape # create 50 rows 2 columns
X[:25,0]+=3
X[:25,1]-=4
kmeans=KMeans(n_clusters=2,random_state=2,n_init=20).fit(X)
#n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，
# 算法默认会初始10次质心，实现算法，然后返回最好的结果;
kmeans.labels_ #输出分类标签 用数字表示的标签
X
fig, ax = plt.subplots(figsize=(4,4))
plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, alpha=0.5)
ax.set_title ('Kmeans clustering results with k=2')

kmeans=KMeans(n_clusters=3,random_state=3,n_init=20).fit(X)
fig, ax = plt.subplots(figsize=(4,4))
plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, alpha=0.5)
plt.title ('Kmeans clustering results with k=2')
kmeans.inertia_

kmeans1=KMeans(n_clusters=3,random_state=3,n_init=1).fit(X)
kmeans1.inertia_ # total sum of squares between points and centroid

#Hierarchical cluster
Xcomplete = AgglomerativeClustering (distance_threshold=0,n_clusters=None, linkage='complete').fit(X)

Xaverage = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average').fit(X)

Xsingle = AgglomerativeClustering(distance_threshold=0,n_clusters=None, linkage='single').fit(X)

#cargs={'color_threshold':-np.inf,'above_threshold_color':'black'}
linkage_compute = compute_linkage (Xcomplete) 
# linkage计算函数从ISLP中导入的
dendrogram(linkage_compute) # **cargs  add this to remove color
cut_tree (linkage_compute, n_clusters=4)
cut_tree (linkage_compute, height=5)

# standardize 
X_scaled = StandardScaler().fit_transform(X)
Xcomplete_scaled = AgglomerativeClustering (distance_threshold=0,n_clusters=None, linkage='complete').fit(X_scaled)
linkage_compute_scaled =compute_linkage (Xcomplete_scaled)
dendrogram(linkage_compute_scaled)


# KMeans
np.random.seed(0)
X=np.random.standard_normal ((30,3))

cor_distance=1-np.corrcoef(X) #X 逐行相关性距离计算
cor_distance.shape # 30乘30的矩阵相关系数
cor_cluster = AgglomerativeClustering (distance_threshold=0,n_clusters=None, linkage='complete', metric='precomputed').fit(cor_distance)
linkage_cor=linkage(cor_distance,method='complete')
dendrogram(linkage_cor)

