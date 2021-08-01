# Assorted Code Snippets

**To check if the variable is of dataframe type** 

PEP8 says explicitly that _isinstance_ is the preferred way to check types

```python
if isinstance(x, pd.DataFrame):
    print("x is dataframe")
```
