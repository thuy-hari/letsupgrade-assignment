#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[5]:


df.head()


# In[6]:


df.columns


# In[8]:


df.info()


# In[9]:


df['Unnamed: 32']


# In[10]:


df = df.drop("Unnamed: 32", axis=1)


# In[11]:


df.head()


# In[12]:


df.columns


# In[13]:


df.drop('id',axis=1, inplace=True)


# In[14]:


df.columns


# In[15]:


type(df.columns)


# In[16]:


l=list(df.columns)


# In[17]:


print(l)


# In[18]:


features_mean=l[1:11]
features_se=l[11:20]
features_worst=l[21:]


# In[19]:


print(features_mean)
print(features_se)
print(features_worst)


# In[20]:


df.head(2)


# In[21]:


df['diagnosis'].unique()
# M= Malignant, B= Benign


# In[22]:


df['diagnosis'].value_counts()


# In[23]:


df.shape


# In[24]:


sns.countplot(df['diagnosis'], label="Count",);


# In[25]:


EXPLORE THE DATA

df.describe()
# summary of all the numeric columns


# In[26]:


df.describe()
# summary of all the numeric columns


# In[27]:


len(df.columns)


# In[28]:


# Correlation Plot
corr = df.corr()
corr


# In[29]:


corr.shape


# In[31]:


plt.figure(figsize=(8,8))
sns.heatmap(corr);


# In[32]:


plt.figure(figsize=(2,2))
sns.heatmap(corr);


# In[33]:


df.head()


# In[34]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
df.head()


# In[35]:


df['diagnosis'].unique()


# In[36]:


X = df.drop('diagnosis', axis=1)
X.head()


# In[37]:


y = df['diagnosis']
y.head()


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[39]:


df.shape


# In[40]:


X_train.shape


# In[42]:


X_test.shape


# In[43]:


y_train.shape
y_test.shape


# In[44]:


X_train.head(1)


# In[46]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[47]:


X_train


# MACHINE LEARNIGN MODELS
# 
# LOGISTIC REGRESSION
# 

# In[48]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[49]:


y_pred = lr.predict(X_test)


# In[50]:


y_pred


# In[51]:


y_test


# In[54]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[55]:


lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[56]:


results = pd.DataFrame()
results


# In[57]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# DECISION TREE CLAISSIFIER
# 

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[59]:


y_pred = dtc.predict(X_test)
y_pred


# In[60]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[61]:


dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)


# In[62]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


#         RANDOM FOREST CLASSIFIER
#         

# In[63]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[64]:


y_pred = rfc.predict(X_test)
y_pred


# In[65]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[66]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[67]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


#     SUPPORT VECTOR CLASSIFIER

# In[68]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[69]:


y_pred = svc.predict(X_test)
y_pred


# In[70]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[72]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[73]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[ ]:





# In[ ]:




