# Essentials:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as plotnine
import seaborn as sns

import gower
from kmodes.kprototypes import KPrototypes
from plotnine import ggplot, geom_line, geom_point, geom_label, labs, xlab, ylab, theme_minimal, aes

# For data visualization
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# for imputation
from sklearn.linear_model import LinearRegression

# Clustering algorithm
from sklearn.cluster import AgglomerativeClustering

# Rand Index
from sklearn.metrics.cluster import rand_score

# Encode labels
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, pairwise_distances
# Confusion Matrix
from sklearn.metrics import confusion_matrix

# DATA LOADING
# //////////////////////////////////////////////////////////////////////////////////////
# TODO this line just makes each output the same - remove it during the verification proces
np.random.seed(42)

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
for i in range(len(data_full.columns)):
    missing_data = data_full[data_full.columns[i]].isna().sum()
    perc = missing_data / len(data_full) * 100
    print('%s,  missing entries: %d, percentage %.2f' % (data_full.columns[i], missing_data, perc))
print("\n")

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

#these are the categorical columns we'll be imputing now
cat_cols = ['Department', 'Term Number', 'Year']

# define the columns to use for imputation
knn_cols = [col for col in data_full_imputed.columns if col not in cat_cols]

# create a copy of the dataframe with only the columns for imputation
data_missing = data_full_imputed[cat_cols].copy()

# impute missing values in the data
imputer = KNNImputer(n_neighbors=5, weights='distance', metric=gower_distance)
imputed_data = pd.DataFrame(imputer.fit_transform(data_full_imputed[knn_cols]))

# combine the imputed data with the original categorical columns
imputed_data.columns = knn_cols
imputed_data.index = data_full_imputed.index
data_imputed = pd.concat([data_full_imputed[cat_cols].fillna('missing'), imputed_data], axis=1)

#This just plots a heatmap to make sure that all values have been successfully imputed
plt.figure(figsize=(6, 5))
sns.heatmap(data_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()

#Code will not continue unless all data has been accurately imputed
assert data_imputed.isna().sum().sum() == 0

#Success! We've finally finished the imputation step.
#We can finally start analyzing it
#We'll use an unsupervised method to make predictions

#Lets use K-prototypes to cluster our data first.

catColumnsPos = [df.columns.get_loc(col) for col in list(data_imputed.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(data_imputed.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

dfMatrix = data_imputed.to_numpy()
# Choose optimal K using Elbow method
cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break
# Converting the results into a dataframe and plotting them
df_cost = pd.DataFrame({'Cluster':range(1, 6), 'Cost':cost})
# Data viz
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_cost)+
    geom_line(aes(x = 'Cluster',
                  y = 'Cost'))+
    geom_point(aes(x = 'Cluster',
                   y = 'Cost'))+
    geom_label(aes(x = 'Cluster',
                   y = 'Cost',
                   label = 'Cluster'),
               size = 10,
               nudge_y = 1000) +
    labs(title = 'Optimal number of cluster with Elbow Method')+
    xlab('Number of Clusters k')+
    ylab('Cost')+
    theme_minimal()
)




# # First, define the number of clusters
# n_clusters = 5
#
# # re-define the categorical columns
# cat_cols = ['Department', 'Term Number', 'Year']
#
# # define the numeric columns
# num_cols = ['Number of Sections', 'Enrollments', 'Median GPA Points', 'Average Section Size', 'Course Number']
#
# # combine the categorical and numeric data into one array
# data_array = np.column_stack((data_imputed[cat_cols], data_imputed[num_cols]))
#
# # define the k-prototypes model
# kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)
#
# # fit the model to the data
# clusters = kproto.fit_predict(data_array, categorical=list(range(len(cat_cols))))
#
# # add the cluster labels to the original data frame
# data_imputed['cluster'] = clusters
#
#
