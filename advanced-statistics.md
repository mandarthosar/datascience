# Advanced Statistics

### Steps for PCA

Import required libraries

```python
#Step 1: Import required packages into Jupyter notebook
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.decomposition import PCA
from statsmodels import multivariate
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
```

Load the file

```python
df = pd.read_csv('file.csv')
```

Quickly check the data

```python
df.shape
df.head() # Shows top 5 rows
df.info()
df.describe() # Use df.describe().T depending on the size of the data
```

Store categorical columns separately so that we can drop them from df dataframe

```python
cat_col1 = df['cat_col1']
cat_col2 = df['cat_col2']
df = df.drop('cat_col1',axis = 1)
df = df.drop('cat_col2',axis = 1)
```

Standardize the dataframe

```python
std_df = pd.DataFrame(zscore(df,ddof=1),columns=df.columns)
np.round(std_df.head(6),2) # Rounds off values to 2 decimal places
```

Principal Component Extraction using sklearn.decomposition package

```python
pca = PCA(n_components=10) # If we are sure that we are going to pick first 10 principle components
pca.fit_transform(std_df)
pc_comps = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'] # values for principle components
prop_var = np.round(pca.explained_variance_ratio_,2) # Proportion of variance
std_dev = np.round(np.sqrt(pca.explained_variance_),2) # Standard deviation
cum_var = np.round(np.cumsum(pca.explained_variance_ratio_),2) # Cumulative variance
temp = pd.DataFrame(pc_comps,columns=['PCs']) 
temp['Proportion Of Variance'] = prop_var 
temp['Standard Deviation'] = std_dev
temp['Cumulative Proportion'] = cum_var
temp
```

Build Screeplot

```python 
plt.figure(figsize=(10,5))
plt.plot(temp['Proportion Of Variance'],marker = 'o')
plt.xticks(np.arange(0,11),labels=np.arange(1,12))
plt.xlabel('# of principal components')
plt.ylabel('Proportion of variance explained')
```

Print first 5 principle components

```python
pc_df_pcafunc = pd.DataFrame(np.round(pca.components_,2),index=pc_comps,columns=std_places.columns)
pc_df_pcafunc.head(5)
```

Find principle components scores

```python
pc = pca.fit_transform(std_df)
pca_df = pd.DataFrame(pc,columns=pc_comps)
np.round(pca_df.iloc[:6,:],2)
```

Correlation matrix of PC scores

```python
round(pca_df.corr(),2)
```

Printing eigenvector and eigenvalues

```python
print("The values for eigen vector:",pca.components_)
print(" ")
print("The values for eigen values:",pca.explained_variance_)
```

Identifying the cumulative eigenvalues

```python
cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)
cum_sum_eigenvalues
```

Identify the loading of features on the principle components

```python
from matplotlib.patches import Rectangle

fig,ax = plt.subplots(figsize=(22, 10), facecolor='w', edgecolor='k')
ax = sns.heatmap(pc_df_pcafunc, annot=True, vmax=1.0, vmin=0, cmap='Blues', cbar=False, fmt='.2g', ax=ax,
                 yticklabels=pc_comps)

column_max = pc_df_pcafunc.abs().idxmax(axis=0)

for col, variable in enumerate(pc_df_pcafunc.columns):
    position = pc_df_pcafunc.index.get_loc(column_max[variable])
    ax.add_patch(Rectangle((col, position),1,1, fill=False, edgecolor='red', lw=3))
```
