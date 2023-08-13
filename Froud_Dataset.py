#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as  plt
import seaborn as sbn
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("C:\\Users\\PCC\\Downloads\\Fraud.csv")


# In[3]:


df


# # 1. Data cleaning including missing values, outliers and multi-collinearity. 
# 

# In[4]:


df.head()


# In[5]:


print(df.info())


# In[6]:


print(df.describe())


# In[7]:


print(df.isnull())


# In[8]:


print(df.isnull().sum())


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


df.dropna()


# In[12]:


df_cleaned = df.dropna()


# In[13]:


Q1 = df_cleaned.quantile(0.25)
 


# In[14]:


Q3 = df_cleaned.quantile(0.75)


# In[15]:


IQR = Q3 - Q1


# In[16]:


df_cleaned = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[17]:


correlation_matrix = df_cleaned.corr()


# In[18]:


plt.figure(figsize=(10, 8))
sbn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# # 2. Describe your fraud detection model in elaboration.

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Data Preprocessing:

# In[20]:


df.head()


# In[21]:


df.fillna(0, inplace=True)


# In[22]:


X = df.drop('isFraud', axis=1)


# In[23]:


y = df['isFraud']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


model = RandomForestClassifier()  # You can choose a different model here


# # 3. How did you select variables to be included in the model?
# 

# In[26]:


X = df.drop('isFraud', axis=1)


# In[27]:


y = df['isFraud']


# In[28]:


correlation_matrix = X.corr()


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


model = RandomForestClassifier()

