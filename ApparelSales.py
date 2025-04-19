#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
df = pd.read_csv(r'C:\Users\harde\Desktop\Data Science Folder\ApparelSales.csv')
df


# In[14]:


print(df.head(20))


# In[7]:


print(df.info())


# In[9]:


print(df.describe())


# In[11]:


print(df.isna().sum())


# In[17]:


kids_sales = df[df['Group'] == 'kids']
print(kids_sales)


# In[19]:


df_cleaned = df.dropna()
df_cleaned.head()


# In[25]:


from sklearn.preprocessing import MinMaxScaler

# Select columns to normalize (assuming 'Sales' and 'Unit')
scaler = MinMaxScaler()
df_cleaned[['Sales', 'Unit']] = scaler.fit_transform(df_cleaned[['Sales', 'Unit']])

df_cleaned


# In[27]:


df_cleaned


# In[29]:


# Descriptive statistics for Sales
print("Sales Statistics:")
print(df_cleaned['Sales'].describe())

# Descriptive statistics for Units
print("\nUnits Statistics:")
print(df_cleaned['Unit'].describe())

# Mean, Median, Mode, and Standard Deviation
sales_mean = df_cleaned['Sales'].mean()
sales_median = df_cleaned['Sales'].median()
sales_mode = df_cleaned['Sales'].mode()[0]  # Mode returns a series, get the first value
sales_std = df_cleaned['Sales'].std()

print(f"\nSales Mean: {sales_mean}")
print(f"Sales Median: {sales_median}")
print(f"Sales Mode: {sales_mode}")
print(f"Sales Standard Deviation: {sales_std}")


# In[33]:


# Group by State
state_sales = df_cleaned.groupby('State')['Sales'].sum().sort_values(ascending=False)
print("State-wise Sales:")
print(state_sales)

# Group by Demographic Group
group_sales = df_cleaned.groupby('Group')['Sales'].sum().sort_values(ascending=False)
print("\nGroup-wise Sales:")
print(group_sales)


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt

# State-wise Sales Analysis
plt.figure(figsize=(5, 3))
sns.barplot(x=state_sales.index, y=state_sales.values)
plt.title('State-wise Sales')
plt.xticks(rotation=30)
plt.show()

# Group-wise Sales Analysis
plt.figure(figsize=(5, 3))
sns.barplot(x=group_sales.index, y=group_sales.values)
plt.title('Group-wise Sales')
plt.xticks(rotation=90)
plt.show()



# In[57]:


# Box Plot for Sales
plt.figure(figsize=(5, 3))
sns.boxplot(y='Sales', data=df_cleaned)
plt.title('Sales Distribution')
plt.show()

# Distribution Plot for Sales
plt.figure(figsize=(5, 3))
sns.histplot(df_cleaned['Sales'], kde=True)
plt.title('Sales Distribution Plot')
plt.show()


# In[ ]:




