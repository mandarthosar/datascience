# Data Mining - KMeans (Non-hierarchical Clustering)

It is an unsupervised Learning algorithm, used to group the unlabeled dataset into different clusters/subsets.

As it is a centroid-based algorithm, ‘means’ in k-means clustering is related to the centroid of data points where each cluster is associated with a centroid.

We classify given data frame through certain number of predetermined clusters or k clusters. 

The performance of the K-means clustering algorithm highly depends upon clusters that it forms. 
Here choosing the optimal number of clusters is critical and difficult. 
There are two popular methods of finding optimal number of clusters i.e. k - Elbow Method and Silhouette score.

**Advantages of using k-means clustering**

* It is easy to implement
* With a large number of variables, K-Means may be computationally faster than hierarchical clustering (if K is small)
* K-Means may produce Higher clusters than hierarchical clustering 

**Disadvantages of using k-means clustering**

* It is difficult to predict the number of clusters (K-Value)
* Initial seeds have a strong impact on the final results

## Working with K-Means

Do the EDA on the daa before this step.

Now, load the required libraries for doing K-Means clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

As K-Means is performed on numerical data only, we have to select only numerical columns. 

```python
df_cluster = 
```

## Elbow Method

The Elbow method is the most popular in finding an optimum number of clusters, this method uses WCSS (Within Clusters Sum of Squares) which accounts for the total variations within a cluster.

WCSS = Addition of all sum of the square of the distances between each data point and its centroid within each cluster

Steps for elbow method:

1. K- means clustering is performed for different values of k (from 1 to 10)
2. WCSS is calculated for each cluster
3. A curve is plotted between WCSS values and the number of clusters k
4. The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K

## Silhouette Score Method

The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 

* If most objects have a high value, then the clustering configuration is appropriate
* If many points have a low or negative value, then the clustering configuration may have too many or too few clusters

