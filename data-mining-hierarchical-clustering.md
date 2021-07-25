# Data Mining

## Hierarchical Clustering

Load libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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
df.isnull().sum()
```

Copy column of interest in new dataframe by filtering or dropping the columns

```python
data_df = df.iloc[:, 1:6] # Filter columns of interest using iloc. These are the continuous variable columns only
```

If scaling is required then do it on the data_df dataframe at this stage. 

Import required libraries for creating dendrogram and linkage

```python
from scipy.cluster.hierarchy import dendrogram, linkage
```

Create an object 'wardlink' using ward method. By default it will use an Eucledian 
The 'linkage' function performs hierarchical/agglomerative clustering.
The method='ward' uses the Ward variance minimization algorithm. The other possible method values are single, complete, average, weighted, centroid

```python
wardlink = linkage(data, method = 'ward')
```

You will want to plot the dendrogram to decide how many clusters you want to choose from the result. 

```python 
dendrogram(wardlink)
dend = dendrogram(wardlink) # If you don't wish to see the complete result and are only interested in dendrogram then use this command instead. When you pass dendrogram() to an object, only plot is shown
```

If you are not interested in the complex dendrogram and instead want to focus on only last couple of clusters then use the following command instead.
Feel free to check p for different numbers, ideally >10 so that you can observe the heights.
The y-axis represents a measure of closeness of either individual data points or clusters.

```python
dend = dendrogram(wardlink,
                 truncate_mode='lastp',
                 p = 20, 
                 )
```

Let's now focus on flattening the result form the dendrogram. fcluster helps us do that. 
It assigns observations to a particular cluster and the result is given in an array format. 
We can then attach the array to original dataframe to study. 

```python
from scipy.cluster.hierarchy import fcluster
```

There are two methods for using fcluster. 
In both methods the result is assigned to the a new array.

Method one focuses on given the count of the maximum clusters we want from this exercise.

```python
#Method 1
clusters = fcluster(wardlink, 3, criterion='maxclust')
clusters
```
Method two focuses on giving the cut-off height of the y-axis on dendrogram.

```python
# Method 2
clusters = fcluster(wardlink, 23, criterion='distance')
clusters
```

Attach the array 'clusters' back to the original dataframe by adding a new column 'Clusters'

```python
df['clusters'] = clusters
```

Check if the original dataframe obtained is looking Ok.

```python
df.head()
```

