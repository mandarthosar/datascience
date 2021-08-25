# EDA

Load libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Distribution of target column

We want to check how the target column observations are distributed.

```python
df['target'].value_counts()
df['target'].value_counts(normalize=True) # Normalized values for target observations
```

## Separating continous and categorical columns

In this step, we will separate out continous data columns and categorical data columns for ease of plotting and further analysis

```python
continuous=df.dtypes[(df.dtypes=='int64')|(df.dtypes=='float64')].index
```

## Univariate analysis

Creating histogram for all columns. You can change the value of bins to define how many bins you want. 

```python
df_bank.hist(bins=10, figsize=(15, 10));  # column='col' - this is an optional parameter
```

Creating blox plot for continous columns

```python
data_plot=df[continuous]
data_plot.boxplot(figsize=(15,10), rot=45);
```

## Bivariate analysis

## Multivariate analysis

We will create pairplot for all continuous variables. 

```python
sns.pairplot(df[continuous])
plt.show()
```
