# Data Mining

## Decision Tree Analysis Classification and Regression Trees or CART 

Load libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

Load decision tree library

```python
from sklearn.tree import DecisionTreeClassifier
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

Decision tree model can accept only numerical or categorical columns. It cannot accept string or object type columns.
The following code will loop through each column and convert any object type column into categorical column. It also assigns distinct category code to the observations. 

```python 
for feature in df.columns:
    if df[feature].dtype == object:
        df[feature] = pd.Categorical(df[feature]).codes
```

Confirm that there's no object column now in the df

```python
df.info()
```

Separate out the target column into a new vector for training set and test set. 
In stats, matrices are denoted by capital letters and vectors are denoted by small letters. Hence, input columns are referred as X because it is a matrix, while target column is referred as y because it is a vector.
Here we are creating new dataframe and series

```python
X = df.drop("target", axis=1)
y = df.pop("target")
```

After we have split the data in input (or independent) and target (or dependent), we now have to split it into training and test data. Hence, an original dataframe would be split in 4 datasets: input training & test and target training & test. 
We will first need to import the library which will allow us to train and test the model
Declaring test size of 30% through _test_size=0.30_

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
```

**Creating Decision Tree model**

Here we are using gini method to calculate the gain. There is another method for calculating gain i.e. entropy. 

```python
dt_model = DecisionTreeClassifier(criterion='gini')
```

Fitting the input and target training observations on decision tree model
```python
dt_model.fit(X_train, y_train)
```

**Now we will create a tree.**

First import required library
We will create labels for binary values of tree splits
We will open a dot file for storing tree data
We will have to close the open file

```python
from sklearn import tree
train_char_label = ['No', 'Yes']
tree_file = open('tree.dot', 'w')
dot_data = tree.export_graphviz(dt_model, out_file=tree_file, feature_names=list(X_train), class_names=list(train_char_label))
tree_file.close()
```

We will list down the importance of each feature.
The importance of each feature refers to total reduction of the criterion brought by that feature. It is also known as Gini importance.
```python
print(pd.DataFrame(dt_model.feature_importances_, columns=["Imp"], index=X_train.columns))
```

We are now trying to predict the values of target column based on X_test part of the data.
```python
y_predict = dt_model.predict(X_test)
```

**Regularizing the decision tree** 

This is important step as it will help control the scope of decision tree. It also makes it little easy to interpret the decision tree decisions at different steps i.e. it improves explainability of the model.

This step exactly like the earlier one with little difference of parameter hypertuning.

```python
reg_dt_model = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_leaf=10, min_samples_split=30)
reg_dt_model.fit(X_train, y_train)
```

We will repeat the above steps for regularized decision tree model. You would notice that the values of importance for different columns have changed due to parameter tuning.

```python
train_char_label = ['No', 'Yes']
reg_tree_file = open('tree.dot', 'w')
dot_data = tree.export_graphviz(dt_model, out_file=reg_tree_file, feature_names=list(X_train), class_names=list(train_char_label))
reg_tree_file.close()
print(pd.DataFrame(reg_dt_model.feature_importances_, columns=["Imp"], index=X_train.columns))
```

We will now predict the train and test data based on the regularized decision tree model using input columns. 

```python
y_train_predict = reg_dt_model.predict(X_train)
y_test_predict = reg_dt_model.predict(X_test)
```

** AUC and ROC for the training data**

Area under the curve and receiver operating characteristic (ROC) curve are important for analyzing the decision tree. ROC is a graph showing the performance of a classification model at all classification thresholds.

For this graph, we have to consider the _probabilities_ of the predicted values.

probs variable gives us an array of negative and positive probabilities.

```python
probs = reg_dt_model.predict_proba(X_train)
probs = probs[:, 1]  # We want to consider the probabilities of the positive outcomes only
```

Calculating area under the curve (AUC)

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_train, probs)
print("Auc: %.3f" %auc)
```

Calculating ROC curve

fpr = false positive rate <br>
tpr = true positive rate 


```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, probs)
```

Plotting lines

```python
plt.plot([0,1], [0,1], linestyle='--'); # Plotting a dashed line between two points 0,0 and 1,1
plt.plot(fpr, tpr, marker='.'); # Plotting a line with fpr on x-axis and tpr on y-axis
plt.xlabel('FPR');
plt.ylabel('TPR');
plt.title('ROC curve for training data');
plt.show()
```

**AUC and ROC for test data**

Repeat above steps for test data.

```python
probs_test = reg_dt_model.predict_proba(X_test)
probs_test = probs_test[:, 1]  # We want to consider the probabilities of the positive outcomes onlyauc = roc_auc_score(y_test, probs)
auc = roc_auc_score(y_test, probs_test)
print("Auc: %.3f" %auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
plt.plot([0,1], [0,1], linestyle='--'); # Plotting a dashed line between two points 0,0 and 1,1
plt.plot(fpr, tpr, marker='.'); # Plotting a line with fpr on x-axis and tpr on y-axis
plt.xlabel('FPR');
plt.ylabel('TPR');
plt.title('ROC curve for test data')
plt.show()
```

**Creating Classification report and Confusion matrix**

Classification reprot for training and test data
Use the print function to have nicely displayed data output. Without print function the data is not easily readable. 

```python
from sklearn.metrics import classification_report, confusion_matrix
print('\n\nClassification report for training data\n')
print(classification_report(y_train, y_train_predict))
print('\n\nClassification report for test data\n')
print(classification_report(y_test, y_test_predict))
```

Confusion matrix for training and test data

```python
print('\n\nConfusion matrix for training data\n')
print(confusion_matrix(y_train, y_train_predict))
print('\n\nConfusion matrix for test data\n')
print(confusion_matrix(y_test, y_test_predict))
```

**Calculating model score**

This score will quickly tell you how performant is your model on training and test data. Theorotically, there will some drop in the test score as model is trying to predict unknown observations with what it has learnt from training data. 

```python
print('Performance of regularized decision tree model on training data:', reg_dt_model.score(X_train, y_train))
print('Performance of regularized decision tree model on test data:', reg_dt_model.score(X_test, y_test))
```
