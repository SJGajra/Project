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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


df=pd.read_csv('data-banknote-authentication.csv',header=None)
df.columns = ['var', 'skew', 'curt', 'entr', 'auth']
df


# In[4]:


df.drop([0,1,2],axis=0,inplace=True)


# In[5]:


df.head(5)


# In[6]:


df.reset_index(drop=True,inplace=True)


# In[7]:


df


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.dtypes


# In[12]:


df=df.astype('float')


# In[13]:


df.dtypes


# In[14]:


sns.pairplot(df,hue='auth')


# In[15]:


plt.figure(figsize=(10,6))
plt.title('Distribution of Target', size=15)
sns.countplot(x=df['auth'])
target_count = df['auth'].value_counts()
plt.annotate(target_count[0],xy=(-0.05 , 10+target_count[0]),size=14)
plt.annotate(target_count[1],xy=(0.9 , 10+target_count[1]),size=14)
plt.ylim(0,900)
plt.show()


# In[16]:


df_to_delete = target_count[0] - target_count[1]
df_to_delete


# In[17]:


df = df.sample(frac=1, random_state=42).sort_values(by='auth')


# In[18]:


df


# In[19]:


df=df[df_to_delete:]


# In[20]:


df


# In[21]:


df['auth'].value_counts()


# In[22]:


x=df.iloc[:,:-1]
x


# In[23]:


y=df.iloc[:,-1]


# In[24]:


y


# In[25]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)


# In[26]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)


# In[27]:


xtest=sc.transform(xtest)


# In[28]:


lr=LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
lr.fit(xtrain,ytrain)


# In[29]:


ypred=lr.predict(xtest)


# In[30]:


train = lr.score(xtrain, ytrain)
test = lr.score(xtest, ytest)
print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")


# In[31]:


print(classification_report(ytest,ypred))


# In[32]:


print(confusion_matrix(ytest,ypred))


# In[33]:


newbanknote = np.array([4.5, -8.1, 2.4, 1.4], ndmin=2)


# In[34]:


newbanknote


# In[35]:


newbanknote=sc.transform(newbanknote)


# In[36]:


print(lr.predict(newbanknote)[0])


# In[37]:


lr.predict_proba(newbanknote)[0]


# In[38]:


newbanknote1 = np.array([-3.72440, 1.90370, -0.035421, -2.50950], ndmin=2)


# In[39]:


newbanknote1=sc.transform(newbanknote1)


# In[40]:


lr.predict(newbanknote1)[0]


# In[41]:


lr.predict_proba(newbanknote1)[0]


# In[ ]:




