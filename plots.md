#Snippets for Plots

### Adding multiple plots on a signle row

```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
sns.boxplot(x=i, data=df, ax=ax1)
sns.boxplot(x=i, data=data_scaled, ax=ax2)
```
