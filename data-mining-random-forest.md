# Data Mining

## Random Forest Decision Tree

Load the libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Load random forest decision tree

```python
from sklearn.ensemble import RandomForestClassifier
```

Load the CSV file

```python
df = pd.read_csv("file.csv")
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

Decision tree model can accept only numerical or categorical columns. It cannot accept string or object type columns. The following code will loop through each column and convert any object type column into categorical column. It also assigns distinct category code to the observations.

```python
for feature in df.columns:
    if df[feature].dtype == object:
        df[feature] = pd.Categorical(df[feature]).codes
```

Confirm that there's no object column now in the df

```python
df.info()
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

**Ensemble RandomForest Classifier**

Creating ensemble for random forest classifier in a standard way.

```python
rfcl = RandomForestClassifier(n_estimators = 501)
rfcl = rfcl.fit(X_train, y_train)
```

Creating ensemble for random forest classifier using cross-validation.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [7, 10],   # Provide values for depth of the tree
    'max_features': [4, 6], # Provide features for the tree
    'min_samples_leaf': [50, 100],  # provide minimum observations in sample leaf
    'min_samples_split': [150, 300],  # Provide value of observations for splitting leaf
    'n_estimators': [301, 501]  # Random numeric value
}

rfcl = RandomForestClassifier() # This will be our estimator input for grid_search

grid_search = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 3) # CV = cross-validation. Here in case we will be doing three set of validations
```

Fitting the model on data.

```python
grid_search.fit(X_train, y_train)
```

Get the best estimator values

```python
grid_search.best_params_
```

Get the best grid

```python
best_grid = grid_search.best_estimator_
```

Getting the y prediction values

```python
y_train_predict = best_grid.predict(X_train)
y_test_predict = best_grid.predict(X_test)
```

Import libraries for confusion matrix and classification report

```python
from sklearn.metrics import confusion_matrix,classification_report
```

** Creating confusion matrix**

Confusion matrix for training data

```python
confusion_matrix(y_train, y_train_predict)
```

Confusion matrix for test data

```python
confusion_matrix(y_test, y_test_predict)
```

**Classification reprot for training and test data**
Use the print function to have nicely displayed data output. Without print function the data is not easily readable.

Classification report for training data

```python
print(classification_report(y_train, y_train_predict))
```

Classification report for test data

```python
print(classification_report(y_test, y_test_predict))
```

**AUC and ROC**

Area under the curve and receiver operating characteristic (ROC) curve are important for analyzing the decision tree. ROC is a graph showing the performance of a classification model at all classification thresholds.

For this graph, we have to consider the _probabilities of the predicted values_.

probs variable gives us an array of negative and positive probabilities.

AUC and ROC for training data

```python

probs = best_grid.predict_proba(X_train)    # Predict probabilities

probs = probs[:, 1]   # Keeping the probabilities for the positive outcome only

# Calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % auc)

# Calculate ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# Plot the ROC curve for the model
plt.plot(fpr, tpr, marker='.')

plt.show()    # Show the plot
```

AUC and ROC for test data

```python

probs = best_grid.predict_proba(X_test)    # Predict probabilities

probs = probs[:, 1]   # Keeping the probabilities for the positive outcome only

# Calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# Calculate ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# Plot the ROC curve for the model
plt.plot(fpr, tpr, marker='.')

plt.show()    # Show the plot
```

**Calculating model score**

This score will quickly tell you how performant is your model on training and test data. Theorotically, there will some drop in the test score as model is trying to predict unknown observations with what it has learnt from training data.

```python
print('Performance of gridsearch model on training data:', reg_dt_model.score(X_train, y_train))
print('Performance of gridsearch model on test data:', reg_dt_model.score(X_test, y_test))
```
