#!/usr/bin/env python
# coding: utf-8

# # Step 0 - Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 1 - Importing dataset

# In[2]:


training_set = pd.read_csv("Train_Titanic.csv")


# In[3]:


training_set.head(5)


# In[4]:


training_set.tail(5)


# # Step 3 - Explore Dataset

# In[5]:


survived = training_set [training_set['Survived'] == 1]
not_survived = training_set [training_set['Survived'] == 0]


# In[6]:


print('Total = ', len(training_set))
print('Number of Passengers survived = ', len(survived))
print('Number of Passengers not survived = ', len(not_survived))
print('% of Passengers survived = ', 1 * len(survived)/ len(training_set) *100)
print('% of Passengers not survived = ', 1 * len(not_survived)/ len(training_set) *100)


# In[7]:


sns.countplot(x = 'Pclass', data = training_set, hue = 'Survived')


# In[8]:


sns.countplot(x = 'Parch', data = training_set, hue = 'Survived')


# In[9]:


sns.countplot(x = 'SibSp', data = training_set, hue = 'Survived')


# In[10]:


sns.countplot(x = 'Embarked', data = training_set, hue = 'Survived')


# In[11]:


sns.countplot(x = 'Sex', data = training_set, hue = 'Survived')


# In[12]:


plt.figure(figsize = (40,30))
sns.countplot(x = 'Age', data = training_set, hue = 'Survived')


# In[13]:


plt.figure(figsize = (40,20))
sns.countplot(x = 'Fare', data = training_set, hue = 'Survived')


# In[14]:


training_set['Age'].hist(bins = 40)


# In[15]:


training_set['Fare'].hist(bins = 40)


# # Step 3 - Prepare data for training/ Data Cleaning

# In[16]:


sns.heatmap(training_set.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[17]:


training_set.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin'], axis = 1, inplace = True)


# In[18]:


sns.heatmap(training_set.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[19]:


plt.figure(figsize = [15,10])
sns.boxplot(x = 'Sex', y = 'Age', data = training_set)


# In[20]:


def fill_age(data):
    age = data[0]
    sex = data[1]
    
    if pd.isnull(age):
        if sex is 'male':
            return 29
        else:
            return 25
    else:
        return age


# In[21]:


training_set['Age'] = training_set[['Age', 'Sex']].apply(fill_age, axis = 1)


# In[22]:


sns.heatmap(training_set.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[23]:


training_set['Age'].hist(bins = 20)


# In[24]:


male = pd.get_dummies(training_set['Sex'], drop_first= True)


# In[25]:


male


# In[26]:


training_set.drop('Sex', axis = 1, inplace = True)


# In[27]:


training_set = pd.concat([training_set, male], axis = 1)


# In[28]:


training_set


# In[29]:


X = training_set.drop('Survived', axis = 1).values


# In[30]:


y = training_set['Survived'].values


# # Step 4 - Model Training

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[32]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# # Step 5 - Model Evaluation

# In[33]:


y_pred1 = classifier.predict(X_validate)


# In[34]:


y_pred1


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report


# In[36]:


cm = confusion_matrix(y_validate, y_pred1)


# In[37]:


sns.heatmap(cm, annot = True)


# In[38]:


report = classification_report(y_validate, y_pred1)
print(report)


# In[ ]:




