# EDA

Load libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Load the data file

```python
df = pd.read_csv('file.csv')
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

## Checking kurtosis

Kurtosis is the sharpness of the peak of a frequency-distribution curve. It is actually the measure of outliers present in the distribution.

* High kurtosis in a data set is an indicator that data has heavy outliers.
* Low kurtosis in a data set is an indicator that data has lack of outliers.
* If kurtosis value is +ve then it means the curve is pointy and —ve means flat.

* Kurtosis > 3: If the distribution is tall and thin it is called a leptokurtic distribution. Values in a leptokurtic distribution are near the mean or at the extremes.
* Kurtosis < 3: A flat distribution where the values are moderately spread out (i.e., unlike leptokurtic) is called platykurtic distribution.
* Kurtosis = 3: A distribution whose shape is in between a leptokurtic distribution and a platykurtic distribution is called a mesokurtic distribution. A mesokurtic distribution looks more close to a normal distribution.

```python
df.kurtosis()
```

## Separating continous and categorical columns

In this step, we will separate out continous data columns and categorical data columns for ease of plotting and further analysis

```python
continuous_cols = df.dtypes[(df.dtypes=='int64')|(df.dtypes=='float64')].index
# continuous_cols = df.select_dtypes(include='number').columns
object_cols = df.select_dtypes(include='object').columns
categorical_cols = df.select_dtypes(include='category').columns




```


## Univariate analysis

Creating histogram for all columns. You can change the value of bins to define how many bins you want. 

```python
df.hist(bins=10, figsize=(15, 10));  # column='col' - this is an optional parameter
```

Creating blox plot for continous columns

```python
data_plot=df[continuous_cols]
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

Creating correlation plot and heatmap

```python
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask = mask, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()
```

Optional heatmap showing correlation with dependent variable

```python
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Target variable', fontdict={'fontsize':18}, pad=16);
```

## Multivariate analysis

We will create pairplot for all continuous variables. 

```python
sns.pairplot(df[continuous_cols])  # hue='target' can be used for better analysis of the target column, height is optional parameter which represent height of scalar in inches
plt.show()
```

We will create scatterplot for all continuous variables.

```python
plt.figure(figsize=(10,10))
sns.scatterplot(data=df[continuous_cols])
plt.show()
```

## Getting unique count of all object columns

```python
for column in object_cols:
    print(column.upper(),': ',df[column].nunique())
    print(df[column].value_counts().sort_values())
    print('\n')
```

## Pandas Profiling

```python
import pandas_profiling
profile = df.profile_report(title='Report on data frame')
profile.to_file(output_file="dataframe-report.html")
```

## Treating Outliars

```python
def replace_outlier(col):
    Q1, Q3 = np.quantile(col, [.25, .75])
    IQR = Q3 - Q1
    LL = Q1 - 1.5*IQR
    UL = Q3 + 1.5*IQR
    return LL, UL

for i in continuous_cols:
    LL, UL = replace_outlier(df[i])
    df_num[i] = np.where(df_num[i]> UL, UL, df_num[i])
    df_num[i] = np.where(df_num[i]< LL, LL, df_num[i])
    
# check if outliars are treated
data_plot=df[continuous_cols]
data_plot.boxplot(figsize=(15,10), rot=45);
```
