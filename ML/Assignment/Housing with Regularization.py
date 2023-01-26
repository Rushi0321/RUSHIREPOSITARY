#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


House_p=pd.read_csv('Housing.csv')


# In[3]:


House_p.head()


# In[4]:


House_p.info()


# In[8]:


House_p.ndim


# In[6]:


dummies=pd.get_dummies(House_p[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']],drop_first=True)


# In[9]:


dummies


# In[10]:


House_p=pd.concat([House_p,dummies],axis=1)


# In[11]:


House_p.head()


# In[15]:


House_p.shape
House_p=House_p.drop(columns=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'],axis=1)


# In[16]:


House_p


# In[17]:


House_p.isnull().sum()


# In[18]:


cor=House_p.corr()


# In[19]:


cor


# In[20]:


correlated_features=set()
for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i,j])>0.50:
            colname1=cor.columns[i]
            colname2=cor.columns[j]
            print(abs(cor.iloc[i,j]),"-",i,"-",j,"-",colname1,"-",colname2)
            correlated_features.add(colname1)
            correlated_features.add(colname1)
            


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
vars=['price','area','bedrooms','bathrooms','stories','parking']
House_p[vars]=scaler.fit_transform(House_p[vars])
House_p


# In[26]:


y


# In[27]:


X


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


x_train.head()


# In[31]:


y_train.head()


# In[32]:


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[36]:


def build_model(x,y):
    x=sm.add_constant(x)
    lm=sm.OLS(y,x).fit()
    print(lm.summary())
    return lm


# In[37]:


model1=build_model(x_train,y_train)


# In[38]:


def checkVIF(X):
    vif=pd.DataFrame()
    vif['features']=X.columns
    vif['VIF']=[variance_inflation_factor(X.values,i) for i in range (X.shape[1])]
    vif['VIF']=round(vif['VIF'],2)
    vif=vif.sort_values(by='VIF',ascending=False)
    return (vif)


# In[39]:


cv=checkVIF(x_train)
cv


# In[42]:


x_train


# In[43]:


x_train1=x_train.drop(['bedrooms','guestroom_yes','furnishingstatus_semi-furnished'],axis=1)


# In[44]:


x_train1


# In[45]:


model2=build_model(x_train1,y_train)


# In[46]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train1,y_train)
rfe=RFE(lm)
rfe=rfe.fit(x_train1,y_train)


# In[47]:


x_train1.columns[rfe.support_]


# In[48]:


x_train_rfe=x_train[x_train1.columns[rfe.support_]]
x_train_rfe


# In[49]:


model3=build_model(x_train_rfe,y_train)
model3


# In[50]:


x_train_rfe=sm.add_constant(x_train_rfe)
x_train_rfe.head()


# In[52]:


y_train_pred=model3.predict(x_train_rfe)
y_train_pred


# In[53]:


x_test.head()


# In[54]:


y_test


# In[55]:


x_test_c=pd.DataFrame(sm.add_constant(x_test))
x_test_c


# In[57]:


y_pred=model1.predict(x_test_c)


# In[58]:


from sklearn.metrics import r2_score
print("test r2 sqrd:",r2_score(y_test,y_pred))


# In[59]:


print("train r2 sqrd:",r2_score(y_train,y_train_pred))


# In[62]:


fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 10)              
plt.xlabel('y_test', fontsize = 10)                          
plt.ylabel('y_pred', fontsize = 10)
plt.show()


# In[63]:


#Regularization>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# In[64]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[65]:


ridge_regressor=Ridge()


# In[67]:


parameters={'alpha':[0.5,1.5,5]}


# In[69]:


ridgecv=GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_absolute_error',cv=5)
ridgecv.fit(x_train,y_train)


# In[70]:


ridgecv.best_params_


# In[71]:


ridgecv.best_score_


# In[72]:


ridge_pred=ridgecv.predict(x_test)


# In[73]:


ridge_pred


# In[74]:


fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs ridge_pred', fontsize = 10)              
plt.xlabel('y_test', fontsize = 7)                          
plt.ylabel('ridge_pred', fontsize = 7)


# In[75]:


score=r2_score(y_test,ridge_pred)
score


# In[ ]:




