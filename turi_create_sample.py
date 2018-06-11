
# coding: utf-8

# # TuriCreate For Titanic

# In[1]:


import turicreate as tc

train =  tc.SFrame('/Users/ninininninnin/.kaggle/competitions/titanic/train.csv')

test =  tc.SFrame('/Users/ninininninnin/.kaggle/competitions/titanic/test.csv')

train.show()


# In[2]:


test['Survived'] = 0
test['IsTest'] = 1
train['IsTest'] = 0
data = train.append(test)


# ## Age

# In[3]:


data['Age'].head(5)


# In[4]:


age_mean = round(data['Age'].mean())
data['Age'] = data['Age'].fillna(age_mean)

age_mean


# In[5]:


data['Age'].show()


# ## Name

# In[6]:


data['Name'].head(5)


# In[7]:


data['InitialName'] = data['Name'].apply(lambda x: x[0])

data['Title'] = data['Name'].apply(lambda x: x.split(', ')[1].split(' ')[0])


# In[8]:


data['InitialName'].show()


# In[9]:


data['Title'].show()


# ## Model

# In[10]:


train_data = data[data['IsTest']==0]
test_data = data[data['IsTest']==1]
model = tc.classifier.create(train_data, target='Survived',
                             features = [
                                         'Pclass',
                                         'InitialName',
                                         'Title',
                                         'Sex',
                                         'Age',
                                         'SibSp',
                                         'Parch',
                                         'Ticket',
                                         'Fare',
                                         'Cabin',
                                         'Embarked'
                             ])


# In[11]:


predictions = model.classify(test_data)

test_data['Survived'] = predictions['class']


# In[12]:


results = model.evaluate(test_data)


# In[13]:


results

