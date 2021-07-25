# Data Mining
## Hierarchical Clustering

Load libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

Import KMeans library

```python
from sklearn.cluster import KMeans
```

Import the data file

```python
df = pd.read_csv("file.csv")
```

Do basic analysis

```python
df.head()
df.describe() 
df.info()
df.shape
df.isnull().sum()
df.duplicated().sum()
```

Copy column of interest in new dataframe by filtering or dropping the columns

```python
data_df = df.iloc[:, 1:6] # Filter columns of interest using iloc. These are the continuous variable columns only
data_df = df.drop(['col1','col2'], axis=1) # Drop the columns and assign the resulting dataframe to new object
```

Check if all the required new columns are present in new dataframe using head()

```python
data_df.head()
```

If scaling is required then do it on the data_df dataframe at this stage.

```python
from sklearn.preprocessing import StandardScaler
X = StandardScaler() # Create an object of StandardScaler
```

Create scaled n-dimensional array scaled_df through fit and transform the 'data_df' which is having columns of our interest and all of them are of continuous type. 

```python
scaled_df = X.fit_transform(data_df) 
```

Check scaled_df array

```python
scaled_df
```

Create k_means object with 2 clusters using KMeans

```python
k_means = KMeans(n_clusters=2)
```

Fit the scaled_df on k_means object
```python
k_means.fit(scaled_df)
```

Check lables of the k_means when we have only two clusters

```python
k_means.labels_
```

Check inertia of the k_means when we have only two clusters
The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares

```python
k_means.inertia_
```

Similarly, you can do it for 3 clusters

```python 
k_means = KMeans(n_clusters=3)
k_means.fit(scaled_df)
k_means.inertia_
```

Create an empty wss[] for capturing within-cluster sum-of-squares at different cluster counts to decide how many clusters would be optimal

```python
wss = []
for i in range(1,11): # Assuming we have to identify optimal cluster count in the range of 1 to 10
    k_means = KMeans(n_clusters = i)
    k_means.fit(scaled_df)
    wss.append(k_means.inertia_)
wss
```

Draw a WSS plot with range(1,11) on x-axis and WSS values for every cluster count on y-axis

```python
plt.plot(range(1,11), wss);
```

Silhouette Score method would be more appropriate for arriving at optimum number of clusters. This is a statistical method and hence should be preferred over visual elbow method. 

```python
sil_score = []
from sklearn.metrics import silhouette_score
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2,12):
  k_means = KMeans(n_clusters = k).fit(scaled_df)
  labels = k_means.labels_
  sil_score.append(silhouette_score(scaled_df, labels, metric = 'euclidean'))
# We are drawing plot for 1 to 10 on x-axis, while the range for k in for loop is 2 to 12. We are trying to maintain the same points on the a-axis.
plt.plot(range(1,11), sil_score);
```

Pick the optimal value of the n_cluster looking at the plot. The count after which the drop is not significant is good to be picked as desired n_cluster value. 
Assuming that value is coming out to be 4, run the following code. Also, we are assigning the resulting labels to labels, which we can add to the original dataframe.
**This step is essential for KMeans as we have to decide the number of clusters beforehand.**

```python
k_means = KMeans(n_clusters=4)
k_means.fit(scaled_df)
labels = k_means.labels_
```

Attach the array 'clusters' back to the original dataframe by adding a new column 'Clusters'. Also, check the newly formed dataframe

```python
df['clusters'] = labels
df.head()
```

Import silhouette libraries for samples and scores

```python
from sklearn.metrics import silhouette_samples, silhouette_score
```

Check silhouette score for the whole result. 
The input is scaled dataframe of continous columns and labels received from optimum n_clusters KMeans
If +ve then points are assigned to each cluster in a proper way
If -ve then points are not assigned to each cluster in a proper way

```python
silhouette_score(scaled_df, labels)
```

You can check the silhouette for each row separately. We will check the minimum value of the silhouette samples to see if any of the row is negative.

```python
silhouette_samples(scaled_df, labels).min()
```

We will add the silhouette sample values for each row to sil_width column which we will add to original dataframe. 

```python
sil_width = silhouette_samples(scaled_df, labels)
df["Sil_Width"] = sil_width
df.head()
```

Check the values of the means of each column for the clusters.

```python 
df.groupby(by="Clusters").mean()
```

Write the output to another CSV file.

```python
df.to_csv("final.csv")
```

