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
plt.xticks(rotation=45)
plt.show()
```

Optional heatmap showing correlation with dependent variable

```python
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Target variable', fontdict={'fontsize':18}, pad=16);
```

Creating barplot between object and continuous data columns

```python
fig=plt.figure(figsize=(20,20))
for i in range(0,len(object_cols)):
    for j in range(0,len(continuous_cols)):
        ax = fig.add_subplot(len(object_cols),len(continuous_cols),i*len(continuous_cols)+j+1)
        sns.barplot(x=object_cols[i], y=continuous_cols[j], data=df)
        ax.set_title("Barplot of "+str(object_cols[i])+" and "+str(continuous_cols[j]),color='Red')
        print(f"i = {i} and j = {j}")
    
plt.tight_layout()
plt.show()
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

## Checking means, std and variance of continous columns graphically

Creating empty lists for storing means, std and variance of the data frame.

```python
df_mean = []
df_std = []
df_var = []
```

Creating a for loop for capturing statistical values for every column.

```python
for i in continuous_cols:
    df_mean.append(df[i].mean())
    df_std.append(df[i].std())
    df_var.append(df[i].var())
```

Creating a data series out of different serieses.

```python
stat_sample = pd.DataFrame(
    {'Means': df_mean,
     'StdDev': df_std,
     'Var': df_var
    })
stat_sample # This will show the values being collected
```

Line plot for stats of data frame

```python
plt.plot(stat_sample)
plt.legend(stat_sample.columns)
```


## Pandas Profiling

```python
import pandas_profiling
profile = df.profile_report(title='Report on data frame')
profile.to_file(output_file="dataframe-report.html")
```

## Treating Outliers

Some models are not affected by outliers. For example, decision tree is not affected by outliers. ANN are not impacted by outliers if their proportion is less i.e. <15%

So, we need to check the proportion of outliers to check if we really need to treat outliers. 

```python
for col in continuous_cols:
    q75,q25 = np.percentile(df[col],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    print("Observations for "+str(col))
    print(f"Count of total observations in a column: {df[col].count()}")
    print(f"Count of outliers: {df[df[col]>max][col].count()+df[df[col]<min][col].count()}")
    print(f"Proportion of outliars in the column: { (df[df[col]>max][col].count()+df[df[col]<min][col].count()) / (df[col].count()) }")
    print()
```

```python
def replace_outlier(col):
    Q1, Q3 = np.quantile(col, [.25, .75])
    IQR = Q3 - Q1
    LL = Q1 - 1.5*IQR
    UL = Q3 + 1.5*IQR
    return LL, UL

for i in continuous_cols:
    LL, UL = replace_outlier(df[i])
    df[i] = np.where(df[i]> UL, UL, df[i])
    df[i] = np.where(df[i]< LL, LL, df[i])
    
# check if outliars are treated
data_plot=df[continuous_cols]
data_plot.boxplot(figsize=(15,10), rot=45);
```

## Scaling the data

General practice for data scaling 

1. Fit the scaler using available training data. This is done by calling the fit() function.
2. Apply the scale to training data. This is done by calling the transform() function.
3. Apply the scale to data going forward.


**Normalization**

Formula = (x – min) / (max – min)

Assumptions:

* Normalization requires that you know or are able to accurately estimate the minimum and maximum observable values.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # Define min max scaler
df_scaled = scaler.fit_transform(data) # Transform data
df_scaled
```

**Standardization**

Formula = (x – mean) / standard_deviation

Subtracting the mean from the data is called _centering_, whereas dividing by the standard deviation is called _scaling_. As such, the method is sometime called "_center scaling_".

The mean and standard deviation estimates of a dataset can be more robust to new data than the minimum and maximum.

Assumptions:

* Standardization requires that you know or are able to accurately estimate the mean and standard deviation of observable values.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   # Define standard scaler
df_scaled = scaler.fit_transform(df) # Transform data
df_scaled
```
The output would be numpy.ndarray. So you may need to convert it back to dataframe using

```python
df_scaled = pd.DataFrame(df_scaled)
```

