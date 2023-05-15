# Essentials:
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import pairwise_distances

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

df = pd.read_csv("C:/Users/allur/Downloads/18-22 datasets (.csv & .xlsx)/national20212022_PUBLIC_instchars.csv", low_memory=False, encoding='mac_roman', na_values = [' '], on_bad_lines= 'skip')
print(df.head())
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))

df.info() #tells us that 14 of the dtypes are float64s. The remaining 1667 are objects
print(df.nunique())

#the first three columns aren't super useful - we can remove them
df = df.drop(['StartDate', 'RecordedDate', 'responseid'], axis = 1)


print(df.info()) #We've correctly dropped the Start Date, Recorded Date, and Response Id from the dataframe

#distance_matrix = gower.gower_matrix(df)

#Not enough memory to use gower.gower_matrix(df). Let's try chunking to sidestep this problem

chunk_size = 1000  # Number of rows to process in each chunk
m = df.shape[0]  # Total number of rows in the dataset

# Initialize an empty sparse distance matrix
distance_matrix = lil_matrix((m, m))

# Iterate over the chunks
for i in range(0, m, chunk_size):
    # Select a chunk of the dataset
    chunk = df.iloc[i:i + chunk_size]

    # Calculate the distance matrix for the current chunk
    chunk_distance_matrix = gower.gower_matrix(chunk)

    # Convert the chunk distance matrix to a sparse matrix
    chunk_distance_matrix_sparse = lil_matrix(chunk_distance_matrix)

    # Update the distance matrix with the values from the current chunk
    distance_matrix[i:i + chunk_distance_matrix.shape[0], i:i + chunk_distance_matrix.shape[1]] = chunk_distance_matrix_sparse

# distance_matrix = pairwise_distances(df, metric='gower')
#
# # Convert distance matrix to sparse matrix
# sparse_matrix = csr_matrix(distance_matrix)