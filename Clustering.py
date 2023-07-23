#!/usr/bin/env python
# coding: utf-8

# # Import software libraries and load the dataset #

# In[1]:


import sys                                             # Read system parameters.
import os                                              # Interact with the operating system.
import numpy as np                                     # Work with multi-dimensional arrays and matrices.
import pandas as pd                                    # Manipulate and analyze data.
import matplotlib as mpl                               # Create 2D charts.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yellowbrick                                     # Visualize elbow and silhouette plots.
import sklearn                                         # Perform data mining and analysis.

# Summarize software libraries used.
print('Libraries used in this project:')
print('- Python {}'.format(sys.version))
print('- NumPy {}'.format(np.__version__))
print('- pandas {}'.format(pd.__version__))
print('- Matplotlib {}'.format(mpl.__version__))
print('- Yellowbrick {}'.format(yellowbrick.__version__))
print('- scikit-learn {}\n'.format(sklearn.__version__))

# Load the dataset.
PROJECT_ROOT_DIR = "."
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "wholesale_customers_data")
print('Data files in this project:', os.listdir(DATA_PATH))
data_raw_file = os.path.join(DATA_PATH, 'wholesale_customers_data.csv')
data_raw = pd.read_csv(data_raw_file)
print('Loaded {} records from {}.'.format(len(data_raw), data_raw_file))


# # Get acquainted with the dataset

# In[2]:


# View data types and see if there are missing entries.
print('Data Types:\n', data_raw.dtypes)
print('\nMissing Entries:\n', data_raw.isnull().sum())

# View first 10 records.
print('\nFirst 10 Records:')
print(data_raw.head(10))


# # Examine the distribution of various features

# In[3]:


# Use Matplotlib to plot distribution histograms for all features.
plt.figure(figsize=(12, 8))
data_raw.hist(bins=20, color='skyblue', edgecolor='black', grid=False)
plt.tight_layout()
plt.show()


# # Examine a general summary of statistics

# In[4]:


# View summary statistics (mean, standard deviation, min, max, etc.) for each feature.

# View summary statistics for each feature
summary_statistics = data_raw.describe()
print(summary_statistics)


# # Use a *k*-means model to label every row in the dataset

# In[5]:


from sklearn.cluster import KMeans

# Create a k-means clustering object with 3 initial clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Use fresh products and milk products only for the initial training data
training_data = data_raw[['Fresh', 'Milk']]

# Fit the training data to the clustering object
kmeans.fit(training_data)

# Predict the cluster labels based on the training data
data_raw['Cluster_Labels'] = kmeans.predict(training_data)


# # Attach cluster labels to the original dataset

# In[6]:


# Append the cluster labels to a new column in the original dataset
data_raw['Cluster_Labels'] = kmeans.labels_

# Show a preview of rows in the dataset with cluster labels added
print(data_raw.head())


# # Show clusters of customers based on fresh products and milk products sales

# In[7]:


# Import seaborn library for customizing the plot style
import seaborn as sns

# Create a scatter plot of customer data
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_raw['Fresh'], data_raw['Milk'], c=data_raw['Cluster_Labels'], cmap='viridis', alpha=0.7)

# Add labels and title
plt.xlabel('Fresh Products')
plt.ylabel('Milk Products')
plt.title('Clusters of Customers based on Fresh and Milk Products Sales')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Customize plot style using seaborn
sns.set_style('whitegrid')
sns.despine()

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Labels')

# Show the plot
plt.show()


# # Use the elbow method to determine the optimal number of clusters

# In[8]:


# Set new DataFrame 'X' equal to the full dataset (all features).
X = data_raw.drop('Cluster_Labels', axis=1)

# Import the necessary modules from Yellowbrick
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Instantiate the KMeans model with the range of clusters from 1 to 10
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))

# Fit the visualizer to the full dataset
visualizer.fit(X)

# Display the plot
visualizer.poof()


# # Use silhouette analysis to determine the optimal number of clusters

# In[9]:


from sklearn.metrics import silhouette_score

# Initialize lists to store silhouette scores and number of clusters
silhouette_scores = []
num_clusters_list = range(2, 6)

# Loop through different number of clusters and calculate silhouette scores
for num_clusters in num_clusters_list:
    model = KMeans(n_clusters=num_clusters)
    labels = model.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Number of Clusters: {num_clusters}, Silhouette Score: {silhouette_avg:.4f}")

# Find the number of clusters with the highest silhouette score
optimal_num_clusters = num_clusters_list[silhouette_scores.index(max(silhouette_scores))]
print(f"\nOptimal Number of Clusters: {optimal_num_clusters}")


# # Generate and preview cluster labels using the full dataset

# In[10]:


from sklearn.cluster import KMeans

# Construct the k-means clustering model
desired_num_clusters = 3
kmeans_model = KMeans(n_clusters=desired_num_clusters, random_state=42)

# Fit the full training data to the model
kmeans_model.fit(X)

# Predict cluster labels for the full dataset
cluster_labels = kmeans_model.predict(X)

# Append cluster labels to the original dataset
data_with_clusters = data_raw.copy()
data_with_clusters['Cluster'] = cluster_labels

# Show the first 20 rows of the dataset with cluster labels added
print(data_with_clusters.head(20))


# In[ ]:




