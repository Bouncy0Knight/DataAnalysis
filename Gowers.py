# Essentials:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as plotnine
import seaborn as sns

import gower
from kmodes.kprototypes import KPrototypes
from plotnine import ggplot, geom_line, geom_point, geom_label, labs, xlab, ylab, theme_minimal, aes, theme

# For data visualization
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# for imputation
from sklearn.linear_model import LinearRegression

# Clustering algorithm
from sklearn.cluster import AgglomerativeClustering, KMeans

# Rand Index
from sklearn.metrics.cluster import rand_score

# Encode labels
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, pairwise_distances
# Confusion Matrix
from sklearn.metrics import confusion_matrix

# DATA LOADING
# //////////////////////////////////////////////////////////////////////////////////////
# this line just makes each output the same - useful during the verification process
np.random.seed(42)

#Read in the location of the file here!
data_full = df = pd.read_csv('C:/Users/allur/PycharmProjects/pythonProject3/Dartmouth - Courses.csv')

# Drops the first column of data in the courses csv
data_full = data_full.iloc[:, 1:]  # the first column (randomly assigned id) won't tell us anything useful
data_full.head()

print("NUMBER OF UNIQUE VALUES:\n", data_full.nunique(), "\n")  # outputs number of different values in each section
# If any of the categories only have 1 possibility (or as many possibilities as entries) we remove them!

# info about the data types stored in the csv
print("DATA INFO:")
data_full.info()  # It looks like we have missing data
print("\n")

print("MISSING ENTRY INFO")
#this should quantize the amount of missing data we have
for i in range(len(data_full.columns)):
    missing_data = data_full[data_full.columns[i]].isna().sum()
    perc = missing_data / len(data_full) * 100
    print('%s,  missing entries: %d, percentage %.2f' % (data_full.columns[i], missing_data, perc))
print("\n")
#it looks like we're missing 10% of the data

# HeatMap of missing values to visualize where we have holes in data
plt.figure(figsize=(6, 5))
# maps the missing values in dataframe to yellow bars
sns.heatmap(data_full.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()  # displays the plot
# Each yellow line is a missing data point. This lets us visualize where we have holes in the data quickly

# IMPUTATION
# ///////////////////////////////////////////////////////////////////////////////////////////
# Split the data into two sets: one with missing values and one without
data_missing = data_full[data_full.isna().any(axis=1)]
data_complete = data_full.dropna()

# Identify the columns to apply each imputation technique to
categorical_cols = ['Department', 'Term Number', 'Year']
knn_cols = ['Number of Sections', 'Enrollments', 'Median GPA Points', 'Average Section Size', 'Course Number']
all_cols = ['Number of Sections', 'Enrollments', 'Median GPA Points', 'Average Section Size', 'Course Number','Department', 'Term Number', 'Year' ]
# Imputation using K-nearest neighbors
imputer = KNNImputer(n_neighbors=5)
imputer.fit(data_full[knn_cols])
# Uses knn for only the columns we specified earlier
data_full_imputed = pd.DataFrame(imputer.transform(data_full[knn_cols]), columns=knn_cols)

# Combine imputed data with original data
# since we don't want to use KNN with Department or Course Number, those values are taken directly from our original df
data_full_imputed[['Department', 'Term Number','Year']] = data_full[['Department', 'Term Number', 'Year']]

# Plot heatmap of imputed data. Lets us visually check that the values we want to be imputed have been imputed
plt.figure(figsize=(6, 5))
sns.heatmap(data_full_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()
# Since we no longer have missing values in the first 4 columns, we know the imputation worked as expected
# However, we still need to make sure that the values we imputed are reasonable. We can do that visually.

# feature name and column index in the data frame
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
# Plots all the new data (that includes imputed data) against the original data
# If imputation is reasonable, then the distribution of the new data should match the distribution of original data
for i, col in enumerate(knn_cols):
    feature_name = col
    feature_index = data_full.columns.get_loc(feature_name)

    # extract the feature values from the complete data set and the imputed data set
    feature_complete = data_complete.iloc[:, feature_index].values
    feature_imputed = data_full_imputed[feature_name].values

    # plot the distributions of the feature before and after imputation
    ax = axs[i // 2][i % 2]
    ax.hist(feature_complete, bins=20, alpha=0.5, label='Complete')
    ax.hist(feature_imputed, bins=20, alpha=0.5, label='Imputed')
    ax.legend(loc='upper right')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
# The distributions match! We can be more confident that our imputations were reasonable


# Now we need to consider how to impute the remaining columns (categorical_cols)

# Let's use gower's distance -- it's ideal for finding distances between categorical variables
# This function just returns the distance matrix between two data frames, using the imported function gower.gower_matrix
def gower_distance(X, Y=None, w=None):
    return gower.gower_matrix(X, Y, w)

# These are the categorical columns we'll be imputing now
cat_cols = ['Term Number', 'Year', 'Department']

# Define the columns to use for imputation
knn_cols = [col for col in data_full_imputed.columns if col not in cat_cols]

# Create a copy of the dataframe with only the columns for imputation
data_missing = data_full_imputed[cat_cols].copy()

#To impute categorical variables, we need to transform them somehow to make them manageable
#I've decided to use one-hot encoding for this purpose, but there are many different options for this step

# Preprocess categorical columns with one-hot encoding
encoder = OneHotEncoder()
encoded_cats = encoder.fit_transform(data_missing)
encoded_cats_df = pd.DataFrame(encoded_cats.toarray(), columns=encoder.get_feature_names_out(cat_cols))

# Impute missing values in the encoded data
imputer = KNNImputer(n_neighbors=5, weights='distance', metric=gower_distance)
imputed_encoded_data = pd.DataFrame(imputer.fit_transform(encoded_cats_df))

# Combine the imputed encoded data with the original categorical columns
imputed_encoded_data.columns = encoded_cats_df.columns
imputed_encoded_data.index = data_full_imputed.index

print(data_full_imputed[knn_cols])
print("_______________________________________")
print(imputed_encoded_data)
data_imputed = pd.concat([data_full_imputed[knn_cols], imputed_encoded_data], axis=1)
print("___________________________")
print(data_imputed)

#This just plots a heatmap to make sure that all values have been successfully imputed
plt.figure(figsize=(6, 5))
sns.heatmap(data_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()

#Code will not continue unless all data has been accurately imputed
assert data_imputed.isna().sum().sum() == 0

#Success! We've finally finished the imputation step.
#We can finally start analyzing the data
#We'll use an unsupervised method to make predictions

#Lets use K-prototypes to cluster our data first.

catColumnsPos = [df.columns.get_loc(col) for col in list(data_imputed.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(data_imputed.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

dfMatrix = data_imputed.to_numpy()
cost = []

#Elbow method: This clusters the dataset 9 times, using values from 1 to 10 as the number of clusters
#It'll calculate the associated cost with each cluster and let us know what the optimal number of clusters are
for cluster in range(1, 10):
    kmean = KMeans(n_clusters=cluster, init='random', random_state=0, n_init = 10)
    kmean.fit(dfMatrix)
    cost.append(kmean.inertia_)
    print('K-means cost is: ' + str(kmean.inertia_))
    print('Cluster initiation: {}'.format(cluster))

# Converting the results into a dataframe and plotting them
print(cost)
df_cost = pd.DataFrame({'Cluster': range(1, 10), 'Cost': cost})

# Data viz
plot = (
    ggplot(data=df_cost) +
    geom_line(aes(x='Cluster', y='Cost')) +
    geom_point(aes(x='Cluster', y='Cost')) +
    geom_label(aes(x='Cluster', y='Cost', label='Cluster'), size=10, nudge_y=1000) +
    labs(title='Optimal number of clusters with Elbow Method') +
    xlab('Number of Clusters k') +
    ylab('Cost') +
    theme_minimal() +
    theme(figure_size=(8, 4.8))
)
print(plot)

#Great! By the elbow method, we can graphically see that the optimal number of clusters is 4. (Note, the elbow method
#can be graphically seen as the inflection point).

#Clustering via Kmeans, using 4 as the value for number of clusters
k = 4
kmean = KMeans(n_clusters=k, init='random', random_state=0, n_init=10)
kmean.fit(dfMatrix)
cluster_labels = kmean.labels_
cluster_centroids = kmean.cluster_centers_

#This plots the clusters we've created.
def plotCentroids(feature1, feature2, features_list):
    feature_1_ind = features_list.index(feature1)  # Replace with the index of Feature A
    feature_2_ind = features_list.index(feature2)  # Replace with the index of Feature B

    # Plot the cluster centroids
    plt.scatter(cluster_centroids[:, feature_1_ind], cluster_centroids[:, feature_2_ind], c='red', marker='x',
                s=100)
    plt.xlabel(feature_names[feature_1_ind])
    plt.ylabel(feature_names[feature_2_ind])
    plt.title('Cluster Centroids')
    plt.show()

#The different types of available features
feature_names = data_imputed.columns
print(feature_names)
feature_names = feature_names.tolist()


plotCentroids('Median GPA Points', 'Department_ECON', feature_names) #The model predicts that As the median GPA decreases, the chance that a class is an ECON class increases
plotCentroids('Average Section Size', 'Median GPA Points', feature_names) #The model predicts that as the average section size decreases, the GPA of the class increases
plotCentroids('Number of Sections', 'Average Section Size', feature_names) #The model predicts that as the number of sections increases, the average section size will increase
plotCentroids('Course Number', 'Median GPA Points', feature_names) #The model predicts that low and high course numbers will have higher GPAs than intermediate course numbers
