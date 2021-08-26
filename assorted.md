# Assorted Code Snippets

**To check if the variable is of dataframe type** 

PEP8 says explicitly that _isinstance_ is the preferred way to check types

```python
if isinstance(x, pd.DataFrame):
    print("x is dataframe")
```

## Related to data types

To know the data types of each column of dataframe 
```python
df.dtypes
```

To know the aggregated data types of columns of dataframe 
```python
df.dtypes.value_counts()
```

## Related to EDA

To impute the data with mean, median or mode 

```python
df[df['col']!='?']['col'].astype('int').median()
df.replace(to_replace ="old_value", value ="new_value", inplace=True) # To replace values across the dataframe
df['col'].replace(to_replace ="old_value", value ="new_value", inplace=True) # To replace values only within specific column from a dataframe
```

Converting the data types of columns

If you use just _int_ instead of _np.int64_ in the following example, then it will try to convert it to either int8 or int32 whichever makes the storage size optimal. By giving np.int64, we are overriding the operation.

```python
df['col'] = df['col'].astype(np.int64)
```

## Plotting

Plotting _subplots_ for all columns in the dataframe using _for-loop_

```python
fig=plt.figure(figsize=(20,20))
for i in range(0,len(df.columns)):
    ax=fig.add_subplot(3,3,i+1) # This will create 3x3 plots. If you want more then you can change the rows and columns numbers accordingly
    sns.distplot(df[df.columns[i]],hist=False)
    ax.set_title(df.columns[i],color='Red')
plt.tight_layout()
plt.show()
```

Plotting correlation heatmap

```python
corr = df.corr()
matrix = np.triu(corr) # sets variable to show lower triangle
plt.figure(figsize=(15, 15)) # sets the size of the image
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200), # here n represents the shades
    square=True,
    annot=True, # this controls whether to show correlation values in the cells
    mask=matrix # this controls whether to render half or full matrix
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

## Scaling

It is important to unscale data to interpret it against the original data. You can do it this way.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_unscaled = scaler.inverse_transform(df_scaled)
```

## Random

**Loading CSV file from different directory**

If you have your source data file in different directory then use this

```python
import os
os.chdir('')
```

Rounding off percentages to 2 decimal columns

```python
print("Percentage of something",round(df['col'].value_counts().values[0]/df['col'].count()*100,2),'%') # values[0] picks the required value from the series
```

