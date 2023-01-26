#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as smd
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv("breast-cancer-wisconsin (1).data")
print(df.head())


# In[3]:


df.columns=["id","CT","Size","Shape",
           "Adhesion","CellSize","Nuclei","Chromatin",
           "Nucleoli","Mitoses","Class"]
print(df.head())


# In[4]:


df.info()


# In[5]:


df['Nuclei'].unique()


# In[6]:


df['Nuclei']=df['Nuclei'].replace('?',np.nan)
df['Nuclei']=pd.to_numeric(df['Nuclei'])


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df=df[~np.isnan(df['Nuclei'])]


# In[10]:


df['Nuclei']=df['Nuclei'].astype('int')


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


for i in df['CT'].unique():

    pie_CT= pd.DataFrame(df[df['CT'] == i]['Class'].value_counts())
    pie_CT.plot.pie(subplots=True, labels = pie_CT.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('CT[i]')
    plt.gca().set_aspect('equal')
    plt.show()


# In[14]:


for i in df['Size'].unique():
    pie_Size= pd.DataFrame(df[df['Size'] == i]['Class'].value_counts())
    pie_Size.plot.pie(subplots=True, labels = pie_Size.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('Size')
    plt.gca().set_aspect('equal')
    plt.show()
    


# In[15]:


for i in df['Shape'].unique():
    pie_Shape= pd.DataFrame(df[df['Shape'] == i]['Class'].value_counts())
    pie_Shape.plot.pie(subplots=True, labels = pie_Shape.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('Shape')
    plt.gca().set_aspect('equal')
    plt.show()


# In[16]:


for i in df['Adhesion'].unique():
    pie_Adhesion= pd.DataFrame(df[df['Adhesion'] == i]['Class'].value_counts())
    pie_Adhesion.plot.pie(subplots=True, labels = pie_Adhesion.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('Adhesion')
    plt.gca().set_aspect('equal')
    plt.show()  


# In[17]:


for i in df['CellSize'].unique():
    pie_CS= pd.DataFrame(df[df['CellSize'] == i]['Class'].value_counts())
    pie_CS.plot.pie(subplots=True, labels = pie_CS.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('CellSize')
    plt.gca().set_aspect('equal')
    plt.show()


# In[18]:


for i in df['Nuclei'].unique():
    pie_NUL= pd.DataFrame(df[df['Nuclei'] == i]['Class'].value_counts())
    pie_NUL.plot.pie(subplots=True, labels = pie_NUL.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('Nuclei')
    plt.gca().set_aspect('equal')
    plt.show()


# In[19]:


for i in df['Chromatin'].unique():
    pie_CHRM= pd.DataFrame(df[df['Chromatin'] == i]['Class'].value_counts())
    pie_CHRM.plot.pie(subplots=True, labels = pie_CHRM.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('CHROMATIN')
    plt.gca().set_aspect('equal')
    plt.show()


# In[20]:


for i in df['Nucleoli'].unique():
    pie_NLOLI= pd.DataFrame(df[df['Nucleoli'] == i]['Class'].value_counts())
    pie_NLOLI.plot.pie(subplots=True, labels = pie_NLOLI.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('NUCLEOLI')
    plt.gca().set_aspect('equal')
    plt.show()


# In[21]:


for i in df['Mitoses'].unique():
    pie_MTOS= pd.DataFrame(df[df['Mitoses'] == i]['Class'].value_counts())
    pie_MTOS.plot.pie(subplots=True, labels = pie_MTOS.index.values, autopct='%1.1f%%', startangle= 75)
    plt.title('MITOSSES')
    plt.gca().set_aspect('equal')
    plt.show()


# In[22]:


df.isnull().sum()


# In[23]:


cor=df.corr()
cor


# In[24]:


correlated_features = set()
for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i, j]) > 0.8:
            colname1 = cor.columns[i]
            colname2 = cor.columns[j]
            print(abs(cor.iloc[i, j]), "--", i, '--', j, '--', colname1, '--', colname2)
            correlated_features.add(colname1)
            correlated_features.add(colname2)


# In[25]:


print(cor.columns)
print('------')
print(correlated_features)


# In[26]:


col=['Size','Shape']
df=df.drop(col,axis=1)


# In[27]:


from sklearn.model_selection import train_test_split
X=df.drop(['id'],axis=1)
y=X.pop('Class')


# In[28]:


df1=pd.DataFrame(y)


# In[29]:


df1


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scale1=MinMaxScaler()


# In[32]:


vars=['Class']
df1[vars]=scale1.fit_transform(df1[vars])


# In[33]:


df1[vars]


# In[34]:


y.head()


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(X,df1,test_size=0.3,random_state=100)


# In[36]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train[[ 'CT','Adhesion', 'CellSize', 'Nuclei','Chromatin', 'Nucleoli', 'Mitoses']]=scaler.fit_transform(x_train[[ 'CT', 'Adhesion', 'CellSize', 'Nuclei','Chromatin', 'Nucleoli', 'Mitoses']])
x_train.head()


# In[37]:


plt.figure(figsize = (20,10))
sns.heatmap(x_train.corr(),annot = True)
plt.show()


# In[38]:


from statsmodels.tools import add_constant as add_constant
x_train_constant = add_constant(x_train)
x_train_constant.head()


# In[39]:


x_train.columns


# In[ ]:





# In[40]:


x_train.dtypes


# In[41]:


y_train.dtypes


# In[42]:


cols=x_train_constant.columns
model = smd.Logit(y_train, x_train_constant[cols])
result = model.fit()
result.summary()


# In[43]:


df3=['CellSize','Mitoses']


# In[44]:


x_train_constant.drop(df3,axis=1,inplace=True)


# In[45]:


model2=smd.Logit(y_train,x_train_constant)
result=model2.fit()
result.summary()


# In[46]:


col_model2=['CT','Adhesion','Nuclei','Chromatin','Nucleoli']


# In[47]:


x_train1=x_train_constant[col_model2]


# In[48]:


logreg=LogisticRegression()
logreg.fit(x_train1,y_train)


# In[49]:


y_train_pred=logreg.predict(x_train1)
sklearn.metrics.accuracy_score(y_train,y_train_pred)


# In[50]:


cm=confusion_matrix(y_train,y_train_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['predicted:0','predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='YlGnBu')


# In[51]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[52]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'Missclassifications = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity/Recall or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Precision/Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# In[53]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

lr_probs = logreg.predict_proba(x_train1)
print(lr_probs)

lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_train, lr_probs)

print('Logistic: ROC AUC = %.3f' % (lr_auc))
lr_fpr, lr_tpr, _ = roc_curve(y_train, lr_probs)
plt.plot(lr_fpr, lr_tpr, marker='*')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[54]:


pd.DataFrame(zip(lr_fpr, lr_tpr), columns=('FPR', 'TPR'))


# In[55]:


arr=np.array(y_train)
y_train_1d=arr.flatten(order='C')


# In[56]:


y_train_1d.ndim


# In[57]:


y_train_pred_final = pd.DataFrame({'Class':y_train_1d,'Class_prob':lr_probs})
y_train_pred_final['ID']=y_train.index
y_train_pred_final.head()


# In[58]:


y_train_pred_final['predicted'] = y_train_pred_final.Class_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[59]:


from sklearn import metrics
cm1=metrics.confusion_matrix(y_train_pred_final.Class,y_train_pred_final.predicted)
print(cm1)


# In[60]:


print(metrics.accuracy_score(y_train_pred_final.Class,y_train_pred_final.predicted))


# In[61]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Class_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[62]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix


num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[63]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.xlabel('Probability')
plt.ylabel('Accuracy/Sensitivity/Sepecificity')
plt.show()


# In[64]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Class_prob.map(lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[65]:


metrics.accuracy_score(y_train_pred_final.Class, y_train_pred_final.final_predicted)


# In[66]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.final_predicted )
confusion2


# In[67]:


TP = confusion2[1,1]  
TN = confusion2[0,0] 
FP = confusion2[0,1] 
FN = confusion2[1,0] 


# In[68]:


TP / float(TP+FN)


# In[69]:


TN / float(TN+FP)


# In[70]:


print(FP/ float(TN+FP))


# In[71]:


print (TP / float(TP+FP))


# In[72]:


print (TN / float(TN+ FN))


# In[73]:


confusion = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.predicted )
confusion


# In[74]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[75]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[76]:


from sklearn.metrics import precision_score, recall_score


# In[77]:


precision_score(y_train_pred_final.Class, y_train_pred_final.predicted)


# In[78]:


from sklearn.metrics import precision_recall_curve


# In[79]:


pd.DataFrame(zip(y_train_pred_final.Class, y_train_pred_final.predicted))


# In[80]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Class, y_train_pred_final.Class_prob)


# In[81]:


pd.DataFrame(zip(p, r, thresholds), columns=('Precision', 'Recall', 'thesholds')).head(10)


# In[82]:


plt.plot(r[:-1], p[:-1], "g-")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# In[83]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.xlabel('Thresholds')
plt.ylabel('Precision (Green) / Recall (Red)')
plt.show()


# In[84]:


x_test = x_test[col_model2]
x_test


# In[85]:


scaler = StandardScaler()

x_test[['CT','Adhesion','Nuclei','Chromatin','Nucleoli']] = scaler.fit_transform(x_test[['CT','Adhesion','Nuclei','Chromatin','Nucleoli']])

x_test


# In[86]:


list(zip(x_train1.columns, x_test.columns))


# In[87]:


y_test_pred = logreg.predict(x_test)


# In[88]:


list(zip(y_test_pred[:10], y_test[:10]))


# In[89]:


y_pred_1 = pd.DataFrame(y_test_pred)


# In[90]:


y_pred_1.head()


# In[91]:


y_test_df = pd.DataFrame(y_test)


# In[92]:


y_test_df['ID'] = y_test_df.index


# In[93]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[94]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[95]:


lr_probs_test = logreg.predict_proba(x_test)
lr_probs_test = lr_probs_test[:, 1]
y_pred_final['Class_prob'] = lr_probs_test


# In[96]:


lr_probs_test


# In[97]:


y_pred_final


# In[98]:


y_pred_final['final_predicted'] = y_pred_final.Class_prob.map(lambda x: 1 if x > 0.42 else 0)


# In[99]:


y_pred_final.head()


# In[100]:


metrics.accuracy_score(y_pred_final.Class, y_pred_final.final_predicted)


# In[101]:


confusion2 = metrics.confusion_matrix(y_pred_final.Class, y_pred_final.final_predicted )
confusion2


# In[102]:


TP = confusion2[1,1]  
TN = confusion2[0,0]
FP = confusion2[0,1] 
FN = confusion2[1,0]


# In[103]:


TP / float(TP+FN)


# In[104]:


TN / float(TN+FP)


# In[105]:


#CROSS-VALIDATION


# In[106]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
param_grid={'C':[1,3,5,7]}
clf=GridSearchCV(LogisticRegression(solver='liblinear'),param_grid=param_grid,cv=4,return_train_score=True)
clf.fit(x_train,y_train)


# In[107]:


clf.best_score_


# In[108]:


clf.best_params_


# In[110]:


param_grid={'C':[3]}
clf=GridSearchCV(LogisticRegression(solver='liblinear'),param_grid=param_grid,cv=4,return_train_score=True)
clf.fit(x_train,y_train)


# In[111]:


clf.best_score_


# In[112]:


clf.best_params_

