#!/usr/bin/env python
# coding: utf-8

# #  What Is Market Segmentation? 
# Market segmentation is a marketing term that refers to aggregating prospective buyers into groups or segments with common needs and who respond similarly to a marketing action.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Customer Data.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# # EDA

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df['MINIMUM_PAYMENTS']=df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())


# In[10]:


df['CREDIT_LIMIT']=df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())


# In[11]:


df.isnull().sum()


# In[12]:


df.drop("CUST_ID",axis=1,inplace=True)


# In[13]:


df.columns


# In[14]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap="viridis")


# In[15]:


columns=df.select_dtypes("int64")
for i in columns:
    plt.figure(figsize=(20,10))
    sns.countplot(data=df,x=i)
    plt.xticks(rotation=90)
    plt.show()


# In[16]:


pip install scipy


# In[17]:


from scipy.stats import skew


# In[18]:


for i in df:
    sns.distplot(df[i])
    plt.show()
    print(skew(df[i]))


# In[19]:


plt.figure(figsize=(15,8))
plt.suptitle("Visualizing the Skewness",fontsize=20)
ptno=1
for i in df:
    if ptno<=13:
        ax=plt.subplot(4,4,ptno)
        sns.distplot(df[i],color="navy")
        plt.xlabel(i,fontsize=14)
    ptno+=1
plt.tight_layout()


# In[20]:


sns.pairplot(df)


# In[21]:


plt.figure(figsize=(15,10))
plt.suptitle("Visualizing the distribution",fontsize=20)
ptno=1
for i in df:
    if ptno<=13:
        ax=plt.subplot(4,4,ptno)
        sns.boxplot(df[i],color="yellow")
        plt.xlabel(i,fontsize=14)
    ptno+=1
plt.tight_layout()


# In[22]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sdf=sc.fit_transform(df)


# In[23]:


from sklearn.decomposition import PCA


# In[24]:


pca=PCA(n_components=2)
pcadf=pca.fit_transform(sdf)
pcdf=pd.DataFrame(data=pcadf ,columns=["PCA1","PCA2"])
pcdf


# In[25]:


from sklearn.cluster import KMeans
WCSS = []
range_val = range(1,15)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(sdf))
    WCSS.append(kmean.inertia_)
plt.plot(range_val,WCSS,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()


# In[26]:


kmeans_model=KMeans(4)
kmeans_model.fit_predict(sdf)
pca_df_kmeans= pd.concat([pcdf,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)


# In[27]:


plt.figure(figsize=(10,8))
ax=sns.scatterplot(x="PCA1",y="PCA2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
plt.title("Clustering using K-Means Algorithm")
plt.show()


# In[28]:


clusters = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[df.columns])


# In[29]:


clusters = sc.inverse_transform(clusters)
clusters= pd.DataFrame(data=clusters,columns=[df.columns])
clusters


# In[30]:


cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
cluster_df


# In[31]:


cluster1 = cluster_df[cluster_df["Cluster"]==0]
cluster1


# In[32]:


cluster2 = cluster_df[cluster_df["Cluster"]==1]
cluster2


# In[33]:


cluster3 = cluster_df[cluster_df["Cluster"]==2]
cluster3


# In[34]:


cluster4 = cluster_df[cluster_df["Cluster"]==3]
cluster4


# In[35]:


sns.countplot(x='Cluster', data=cluster_df)


# In[36]:


cluster_df.head()


# In[37]:


x=cluster_df.iloc[:,:-1]
y=cluster_df.iloc[:,-1]


# In[38]:


y.head()


# In[39]:


x.head()


# In[40]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)


# In[41]:


def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    
    train=model.score(xtrain,ytrain)
    test=model.score(xtest,ytest)
    
    print(f"Training accuracy: {train}\nTesting accuracy: {test}\n\n")
    print(classification_report(ytest,ypred))
    return model


# In[42]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# In[43]:


st_x = StandardScaler()
xtrain = st_x.fit_transform(xtrain)
xtest = st_x.transform(xtest)


# In[45]:


dt=mymodel(DecisionTreeClassifier(criterion='entropy'))
d=DecisionTreeClassifier(criterion='entropy')
d.fit(xtrain,ytrain)
ypred=d.predict(xtest)
print(confusion_matrix(ytest,ypred))


# In[46]:


bnb=mymodel(BernoulliNB())


# In[47]:


gnb=mymodel(GaussianNB())


# In[48]:


logreg=mymodel(LogisticRegression(multi_class="ovr"))


# In[49]:


ovr=mymodel(OneVsRestClassifier(LogisticRegression()))


# In[50]:


from sklearn.model_selection import cross_val_score
cvs=cross_val_score(ovr,x,y,cv=5,scoring="accuracy")
print(f"Avg.Accuracy :{cvs.mean()}\nSTD : {cvs.std()}")

