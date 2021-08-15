# Data Mining

## Artificial Neural Network 

Load libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
```

If file is in different directory then only load this

```python
import os
os.chdir("")
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

Remove unnecessary columns

```python
df = df.drop(['col1','col2'], axis=1)
```

Import train-test-split model

```python
from sklearn.model_selection import train_test_split
```

Separate out the target column into a new vector for training set and test set. In stats, matrices are denoted by capital letters and vectors are denoted by small letters. Hence, input columns are referred as X because it is a matrix, while target column is referred as y because it is a vector. Here we are creating new dataframe and series

```python
X = df.drop("target", axis=1)
y = df.pop("target")
```

After we have split the data in input (or independent) and target (or dependent), we now have to split it into training and test data. Hence, an original dataframe would be split in 4 datasets: input training & test and target training & test. We will first need to import the library which will allow us to train and test the model Declaring test size of 30% through test_size=0.30

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
```

Import standard scalar library for standardizing the observations

```python
from sklearn.preprocessing import StandardScaler
```

Create standard scalar object

```python
sc = StandardScaler()
```

Fit the standard scalar model on X_train data

```python
X_train = sc.fit_transform(X_train)
```

We will now transform the X_test data based on model which is fit on X_train data. It is important to just transform the data. 

```python
X_test = sc.transform(X_test)
```

We will now create a classifier object 'clf' from MLPClassifer

```python
clf = MLPClassifier(
    hidden_layer_sizes=100, 
    max_iter=5000,  # maximum iterations
    solver='sgd',  # 'sgd' refers to stochastic gradient descent for weight optimization
    verbose=True,  # for printing progress to console
    random_state=21,
    tol=0.01 # tolerance for the optimization
    )
```

We will now fit the classifier model on data

```python
clf.fit(x_train, y_train)
```

Import library for building confusion matrix and classification report

```python
from sklearn.metrics import confusion_matrix, classification_report
```

Create data for confusion matrix for test data

```python
y_test_pred = clf.predict(X_test)
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test
```

Create data for confusion matrix for train data

```python
y_train_pred = clf.predict(X_train)
cm_test = confusion_matrix(y_train, y_train_pred)
cm_test
```

Print classification report for test data

```python
print(classification_report(y_test, y_test_pred))
```

Print classification report for train data

```python
print(classification_report(y_train, y_train_pred))
```

**Creating AUC score and ROC**

Loading required libraries

```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
```

Creating AUC score and ROC for test data

```python
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
```

Creating AUC score and ROC for train data

```python
# predict probabilities
probs = clf.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC

auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % auc)
# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
```

















