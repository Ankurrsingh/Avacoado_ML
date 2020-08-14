#!/usr/bin/env python
# coding: utf-8

# # githublink:https://github.com/Ankurrsingh/Avacoado_ML

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Avocado.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


import seaborn as sns


# In[9]:


sns.pairplot(df)


# In[10]:


sns.countplot('type',data = df)


# In[11]:


sns.barplot(x = 'year',y = 'Total Volume',data = df,palette= 'Blues')


# In[12]:


sns.heatmap(df.corr())


# In[13]:


sns.boxplot('year','AveragePrice',data = df)


# In[14]:


sns.distplot(df['AveragePrice'])


# In[15]:


sns.lineplot('Date','AveragePrice',hue = 'year',data = df)


# In[16]:


sns.lineplot('Date','Total Volume',hue = 'year',data = df)


# In[17]:


sns.swarmplot('Date','AveragePrice',data = df,hue = 'type')


# In[18]:


sns.catplot('year','Total Volume',data = df)


# In[19]:


sns.catplot('year','Total Bags',data = df)


# In[20]:


df.head(5)


# In[21]:


df=df.drop('Unnamed: 0',axis=1)


# In[22]:


df.head(2)


# In[23]:


df["region"].unique()


# In[24]:


df=df.drop("Date",axis=1)


# In[25]:


df.rename(columns = {'4046':'PUC4046_sold'}, inplace = True) 


# In[26]:


df.rename(columns = {'4225':'PUC4225_sold'}, inplace = True) 


# In[27]:


df.rename(columns = {'4770':'PUC4770_sold'}, inplace = True) 


# In[28]:


df.head(3)


# In[29]:


df["type"].unique()


# In[30]:


from sklearn.preprocessing import LabelBinarizer


# In[31]:


lb=LabelBinarizer()


# In[32]:


df["type"]=lb.fit_transform(df.type)


# In[33]:


df["type"].unique()


# In[34]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["region"]=le.fit_transform(df["region"])


# In[35]:


df.corr()


# In[36]:


sns.heatmap(df.corr())


# In[37]:


sns.lineplot(x='Total Volume',y='AveragePrice',data=df)


# In[38]:


sns.lineplot(x='Total Volume',y='AveragePrice',data=df,hue='year')


# In[39]:


sns.barplot(data=df, x="year", y="Total Volume", hue="type", palette=['blue', 'red'], saturation=0.6)


# In[40]:


sns.barplot(data=df, x="year", y="AveragePrice", hue="type", palette=['blue', 'red'], saturation=0.6)


# In[41]:


plt.scatter(df["AveragePrice"],df['Total Volume'])


# In[42]:


sns.kdeplot(df["AveragePrice"], shade=True)


# In[43]:


sns.kdeplot(df["Total Volume"], shade=True)


# In[44]:


sns.kdeplot(df["type"], shade=True,label = "{} cyl".format(df["type"]))


# In[45]:


for avacadotype in df["type"].unique():
    x = df["type"] == avacadotype
    sns.kdeplot(x, shade=True, label = "{} type".format(avacadotype))


# In[46]:


d = df["type"].value_counts().to_dict()
plt.pie(d.values(),
       labels = d.keys(),
       autopct = '%1.1f%%', 
       textprops = {'fontsize': 10, 'color' : "white"} 
      )


# In[47]:


plt.plot(df["AveragePrice"], color = "red", alpha = .5)


# In[48]:


df.columns


# In[49]:


fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot()
ax.plot(df["PUC4046_sold"], color = "red", alpha = .5)
ax.plot(df["PUC4225_sold"], color = "blue", alpha = .5)
ax.plot(df["PUC4770_sold"], color = "green", alpha = .5)
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)


# In[50]:


fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot()
ax.plot(df["PUC4046_sold"],df["AveragePrice"], color = "red", alpha = .7,label='PUC4046')
ax.plot(df["PUC4225_sold"],df["AveragePrice"], color = "blue", alpha = .5,label='PUC4225')
ax.plot(df["PUC4770_sold"],df["AveragePrice"], color = "yellow", alpha = .9,label='PUC4770')
ax.legend()


# # Importing Modules

# In[51]:


import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# # Defining Model And Evaluating

# In[52]:


classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]


# In[53]:


df.head(3)


# # Defining  The model

# In[54]:


X=df.drop("AveragePrice",axis=1)
Y=df["AveragePrice"]


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=True,train_size=5000,random_state=2)


# In[57]:


X_train


# In[58]:


X_test.shape


# In[59]:


Y_train.shape


# In[60]:


Y_test.shape


# In[61]:


df.info()


# In[62]:


for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    print(Y_pred,'\n')
    print('r2 Score:  ',r2_score(Y_test, Y_pred),'\n')
    print('mean_absolute_error:  ',mean_absolute_error(Y_test, Y_pred),'\n')
    print('mean_scored_error:  ',mean_squared_error(Y_test, Y_pred),'\n')


# In[ ]:




