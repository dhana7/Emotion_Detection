
# coding: utf-8

# In[4]:


import pandas as pd
df=pd.read_csv("C:/Users/SHARADA/Downloads/train.csv")
df.head()


# In[5]:


df.shape


# In[6]:


#Extracting the pixels value
pixels = []
for i in df['Pixels']:
    points = i.split(' ')
    x = []
    for j in points:
        x.append(float(j))
    pixels.append(x)


# In[7]:


#Extracting the target value
target = df['Emotion']


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import model_selection


# In[9]:


x_train,x_test,y_train,y_test = model_selection.train_test_split(pixels,target,test_size = 0.25,random_state = 1)
scaler = preprocessing.MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[10]:


#Naive Bayes
clf_nb = GaussianNB()
#Fitting the model
clf_nb.fit(x_train,y_train)
#Predicting for testing data
y_pred_nb = clf_nb.predict(x_test)
#Printing the accuracy score
print("NB:",metrics.accuracy_score(y_test,y_pred_nb))


# In[12]:


#KNN
clf_knn = neighbors.KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
y_pred_knn = clf_knn.predict(x_test)
print("KNN:",metrics.accuracy_score(y_test,y_pred_knn))


# In[17]:


#Decision Tree
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(x_train,y_train)
y_pred_dt = clf_dt.predict(x_test)
print("DT:",metrics.accuracy_score(y_test,y_pred_dt))


# In[14]:


#SVM with linear Kernel
clf_svm_l = svm.SVC(kernel = 'linear')
clf_svm_l.fit(x_train,y_train)
y_pred_svm_l = clf_svm_l.predict(x_test)
print("SVM Linear:",metrics.accuracy_score(y_test,y_pred_svm_l))


# In[15]:


#SVM with RBF Kernel
clf_svm_r = svm.SVC(kernel = 'rbf')
clf_svm_r.fit(x_train,y_train)
y_pred_svm_r = clf_svm_r.predict(x_test)
print("SVM RBF:",metrics.accuracy_score(y_test,y_pred_svm_r))

