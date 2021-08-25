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

## Checking skewness

Skewness is an important property of the sample. 

Direction of skewness

* A skewness value of 0 in the output denotes a symmetrical distribution
* A positive skewness value in the output deonotes an asymmetry in the distribution and the tail is larger towards the **right hand side** of the distribution
* A negative skewness value in the output deonotes an asymmetry in the distribution and the tail is larger towards the **left hand side** of the distribution

Degree of skewness 

* If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
* If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed
* If the skewness is less than -1 or greater than 1, the data are highly skewed

```python
df.skew()
```


## Separating continous and categorical columns

In this step, we will separate out continous data columns and categorical data columns for ease of plotting and further analysis

```python
continuous=df.dtypes[(df.dtypes=='int64')|(df.dtypes=='float64')].index
```


## Univariate analysis

Creating histogram for all columns. You can change the value of bins to define how many bins you want. 

```python
df.hist(bins=10, figsize=(15, 10));  # column='col' - this is an optional parameter
```

Creating blox plot for continous columns

```python
data_plot=df[continuous]
data_plot.boxplot(figsize=(15,10), rot=45);
```

## Bivariate analysis

In bivariate analysis, we compare two columns of the dataset.

For categorical columns, scatter plot are better representation.

```python
sns.boxplot(x='col1', y='col2', data=df)
plt.show()
```

For continous columns, scatter plot are better representation.

```python
sns.scatterplot(x='col1', y='col2', data=df)
plt.show()
```

## Multivariate analysis

We will create pairplot for all continuous variables. 

```python
sns.pairplot(df[continuous])  # hue='target' can be used for better analysis of the target column, height is optional parameter which represent height of scalar in inches
plt.show()
```
