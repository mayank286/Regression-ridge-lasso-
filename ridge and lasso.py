#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_boston


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df = load_boston()


# In[7]:


df


# In[8]:


dataset = pd.DataFrame(df.data)
print(dataset.head())


# In[9]:


dataset.head()


# In[10]:


dataset.tail()


# In[11]:


dataset.columns=df.feature_names


# In[12]:


dataset.tail()


# In[13]:


dataset.head()


# In[14]:


df.target.shape


# In[15]:


dataset["Price"]=df.target


# In[16]:


dataset.head()


# In[17]:


X=dataset.iloc[:,:-1] ## independent features
y=dataset.iloc[:,-1] ## dependent features


# In[18]:


##linear regression


# In[21]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[22]:


## ridge regression


# In[20]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)


# In[23]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[24]:


## lasso regression


# In[25]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[27]:


prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)


# In[28]:


import seaborn as sns

sns.distplot(y_test-prediction_lasso)


# In[29]:


import seaborn as sns

sns.distplot(y_test-prediction_ridge)


# In[ ]:




