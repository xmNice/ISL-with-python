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
# From ISLP.cluster import compute_linkage

# PCA on NCI60DATA
NCI60 = load_data('NCI60')
type(NCI60) # dict
 NCI60.keys() 

nci_data=NCI60['data']
nci_data.shape
type(nci_data) # array
nci_data[:5,:5]

labels=NCI60['labels']
labels.shape
type(labels) # Dataframe
labels.head()
labels.value_counts()
labels.nunique()

nci_scaled = StandardScaler().fit_transform(nci_data)
nci_pca=PCA()
nci_scores=nci_pca.fit_transform(nci_scaled)

nci_groups=pd.factorize(labels["label"])
# A tuple including 2 arrays: values and index
nci_groups[0] # Not really need this step

sns.relplot(x=nci_scores[:,0], y=nci_scores[:,1],hue=labels["label"])

sns.relplot(x=nci_scores[:,0], y=nci_scores[:,2],hue=labels["label"])

ticks=np.arange(nci_pca.n_components_)+1

sns.pointplot(x=ticks, y=nci_pca.explained_variance_ratio_)
# Max PC= 64  For high dimensional data, if nums of row < nums of cols, max PC = nums of rows - 1
sns.pointplot(x=ticks, y=nci_pca.explained_variance_ratio_.cumsum())
# Change is very limit from 8th point. Analyzing more than 8 pc is not benefit. But the first 8 pcs explain only 50% of variance. 

cum=pd.DataFrame(nci_pca.explained_variance_ratio_.cumsum(),columns=["cum_ration"],index=ticks)
cum.head(10) 

# AgglomerativeClustering
linkage_computed1=linkage(nci_scaled, method='complete', metric='euclidean')
dendrogram(linkage_computed1, labels=np.asarray(labels))

linkage_computed2=linkage(nci_scaled, method='average', metric='euclidean')
dendrogram(linkage_computed2, labels=np.asarray(labels))

linkage_computed3=linkage(nci_scaled, method='single', metric='euclidean')
dendrogram(linkage_computed3, labels=np.asarray(labels))

complete_cut= cut_tree(linkage_computed1, n_clusters=4).reshape(-1)
complete_cut  # From 0 to 3 after cutting   4 clusters
# cut_tree (linkage_computed1, height=140)
# Cut in 140, 4 clusters
# reshape reshape(m,-1) reshape(-1,m) 
new_labels=pd.Series(complete_cut.reshape(-1), name='4 clusters')  # To series
new_labels.head(20)
labels
pd.crosstab(labels['label'], new_labels)
# From crosstab, most of cell lines are correctly clustered, such as LEUKEMIA, but BREAST cell line are dispersed into 3 classes

# Representative plot by PC1 PC2 and new labels
sns.relplot(x=nci_scores[:,0], y=nci_scores[:,1],hue=new_labels, palette='deep')

sns.scatter3D 

fig=plt.figure()
ax=plt.axes (projection='3d')
ax.scatter3D (x=nci_scores[:,0], y=nci_scores[:,1],z=nci_scores[:,2], color='green')

# Seems that this model is useless
complete_model=AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='complete').fit(nci_scaled)

# KMeans
nci_kmean=KMeans(random_state=0, n_clusters=4, n_init=(20)).fit(nci_scaled)
pd.crosstab(new_labels, pd.Series(nci_kmean.labels_, name='km clusters'))

# HC on pcs
linkage_computed4=linkage(nci_scores[:,:5], method='complete', metric='euclidean')
dendrogram(linkage_computed4, labels=np.asarray(labels))

pca_cut= cut_tree(linkage_computed4, n_clusters=4).reshape(-1)
pca_labels=pd.Series(pca_cut.reshape(-1), name='pca clusters') 
pd.crosstab(new_labels, pca_labels)