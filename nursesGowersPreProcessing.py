# Essentials:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import gower

#For data visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Clustering algorithm
from sklearn.cluster import AgglomerativeClustering

# Rand Index
from sklearn.metrics.cluster import rand_score

# Encode labels
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
# Confusion Matrix
from sklearn.metrics import confusion_matrix

#TODO this line just makes each output the same - remove it during the verification proces
np.random.seed(42)

data_full = df = pd.read_csv('C:/Users/allur/PycharmProjects/pythonProject/NursesAnalysis/nursery.data')
data_full.head()
print(data_full.nunique())
#This line removes the rec_status (the answers) from our csv file
target = data_full[['rec_status']]
data_no_target = data_full.drop(['rec_status'],axis=1)
data_no_target.head()

#This line fives us info about the data types stored in the csv. Everything except for children is a categorical var
#data_no_target.info()

#gives us the unique number of possibilities in each category
print(data_no_target.nunique())

#TODO: COME BACK AND ADD NUMERICAL VAR BACK
#I'm removing children because it is the only numerical variable we have (not necessary, but useful on first time through
#to understand what's going on
#data_categorical = data_no_target.drop(['children'], axis=1)
data_categorical = data_no_target


distance_matrix = gower.gower_matrix(data_categorical)

print(distance_matrix)
print("FINISHED")
################################//////////////////////////////////////////////
encoder = preprocessing.LabelEncoder()

encoded_target = target.apply(encoder.fit_transform)

print(f'in this encoding, {encoded_target.iloc[0].values} represents {target.iloc[0].values}')

labels = pd.DataFrame()
labels['target'] = encoded_target.values.reshape(1, -1).tolist()[0]

#///////////////////////////////////////////////////////////////////////////////////

model_single = AgglomerativeClustering(n_clusters=5, linkage='single', metric='precomputed')
clusters_single = model_single.fit_predict(distance_matrix)
labels['single-predictions'] = clusters_single
sri = rand_score(encoded_target.values.reshape(1, -1)[0], clusters_single)
print(f'Rand Index: {sri}')
labels[['single-predictions']].value_counts().plot.pie(autopct='%1.0f%%', pctdistance=0.7, labeldistance=1.1)
plt.show()

silhouette_avg_single = silhouette_score(distance_matrix, clusters_single, metric='precomputed')
print(f'Silhouette Score for Single Linkage: {silhouette_avg_single}')

pca = PCA(n_components=4)
data_pca = pca.fit_transform(distance_matrix)

# create scatter plot of the clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_single)
plt.title("Single Linkage Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
#//////////////////////////////////////////////////////////////////////////////////////
model_average = AgglomerativeClustering(n_clusters=5, linkage='average', metric='precomputed')
clusters_average = model_average.fit_predict(distance_matrix)
labels['average-predictions'] = clusters_average
ari = rand_score(encoded_target.values.reshape(1, -1)[0], clusters_average)
print(f'Rand Index: {ari}')
labels[['average-predictions']].value_counts().plot.pie(autopct='%1.0f%%', pctdistance=0.7, labeldistance=1.1)
plt.show()

silhouette_avg_average = silhouette_score(distance_matrix, clusters_average, metric='precomputed')
print(f'Silhouette Score for Average Linkage: {silhouette_avg_average}')

pca = PCA(n_components=4)
data_pca = pca.fit_transform(distance_matrix)

# create scatter plot of the clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_average)
plt.title("Average Linkage Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
#////////////////////////////////////////////////////////////////////////////////////////////
model_complete = AgglomerativeClustering(n_clusters=5, linkage='complete', metric='precomputed')
clusters_complete = model_complete.fit_predict(distance_matrix)
labels['complete-predictions'] = clusters_complete
cri = rand_score(encoded_target.values.reshape(1, -1)[0], clusters_complete)
print(f'Rand Index: {cri}')
labels[['complete-predictions']].value_counts().plot.pie(autopct='%1.0f%%', pctdistance=0.7, labeldistance=1.1)
plt.show()

silhouette_avg_complete = silhouette_score(distance_matrix, clusters_complete, metric='precomputed')
print(f'Silhouette Score for Complete Linkage: {silhouette_avg_complete}')

pca = PCA(n_components=4)
data_pca = pca.fit_transform(distance_matrix)

# create scatter plot of the clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_complete)
plt.title("Complete Linkage Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
#///////////////////////////////////////////////////////////////////////////////////////////
# model_centroid = AgglomerativeClustering(n_clusters=5, linkage='centroid', metric='precomputed')
# clusters_centroid = model_centroid.fit_predict(distance_matrix)
# labels['centroid-predictions'] = clusters_centroid
# cri = rand_score(encoded_target.values.reshape(1, -1)[0], clusters_centroid)
# print(f'Rand Index: {cri}')
# labels[['centroid-predictions']].value_counts().plot.pie(autopct='%1.0f%%', pctdistance=0.7, labeldistance=1.1)
# plt.show()
#
# silhouette_avg_centroid = silhouette_score(distance_matrix, clusters_centroid, metric='precomputed')
# print(f'Silhouette Score for Complete Linkage: {silhouette_avg_centroid}')
# #///////////////////////////////////////////////////////////////////////////////////////////
# model_median = AgglomerativeClustering(n_clusters=5, linkage='median linkage', metric='precomputed')
# clusters_median = model_median.fit_predict(distance_matrix)
# labels['median-predictions'] = clusters_median
# cri = rand_score(encoded_target.values.reshape(1, -1)[0], clusters_median)
# print(f'Rand Index: {cri}')
# labels[['median-predictions']].value_counts().plot.pie(autopct='%1.0f%%', pctdistance=0.7, labeldistance=1.1)
# plt.show()
#
# silhouette_avg_median = silhouette_score(distance_matrix, clusters_median, metric='precomputed')
# print(f'Silhouette Score for Complete Linkage: {silhouette_avg_median}')
#///////////////////////////////////////////////////////////////////////////////////////////



# labels.value_counts(["target", "complete-predictions"])
# labels['aligned-clusters'] = labels['complete-predictions'].apply(lambda x: int(not x))
# labels.value_counts(["target", "aligned-clusters"])
#
# cf_matrix = confusion_matrix(encoded_target.values.reshape(1, -1)[0], labels[["aligned-clusters"]].values.reshape(1, -1)[0])
# cf_labels = ['True Neg','False Pos','False Neg','True Pos']
# cf_labels = np.asarray(cf_labels).reshape(2,2)
# fig, ax = plt.subplots(1, 1)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=cf_labels, fmt='', cmap='Blues')
# ax.set_ylabel('Target Labels')
# ax.set_xlabel('Predicted Labels')
#
# True_neg = cf_matrix[0,0]
# False_pos = cf_matrix[0,1]
# True_pos = cf_matrix[1,1]
# False_neg = cf_matrix[1,0]
#
# accuracy = (True_neg + True_pos)/(True_neg + False_neg + True_pos + False_pos)
# recall = (True_pos)/(False_neg+True_pos)
# precision = (True_pos)/(False_pos + True_pos)
# F1_score = 2 * ((precision*recall)/(precision+recall))
# print(f'Accuracy: {accuracy}')
# print(f'Recall: {recall}')
# print(f'Precision: {precision}')
# print(f'F1_score: {F1_score}')