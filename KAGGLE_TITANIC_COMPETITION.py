#!/usr/bin/env python
# coding: utf-8

# In[30]:


# modules for reading the data
import pandas as pd
import numpy as np


# In[31]:


#getting the data and loading to pandas DataFrame
train_df = pd.read_csv('F://bizz//DATA SETS/TITANIC DATA SET/train.csv', index_col = 'PassengerId' )
test_df = pd.read_csv('F://bizz//DATA SETS/TITANIC DATA SET/test.csv', index_col = 'PassengerId')


# In[45]:


#to have a feel of training data
train_df.head()


# In[33]:


#to have a feel of test data
test_df.head()


# In[46]:


# Survived will act as our label on the testing data.
labl = train_df[ 'Survived']
type(labl)


# In[35]:


#converting series object to dataframe
y = pd.DataFrame(labl, index = train_df.index)
y.head()


# In[36]:


#available features we only chose numerical
train_df.info()


# # Creating a pipeline for numerical features only

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#transformations on the column ie choosing only numerical features
from sklearn.tree import DecisionTreeClassifier
#algorithm for predicting the results


# In[38]:


# selecting columns to use
columns = ['Pclass', 'Parch', 'SibSp']

ct = ColumnTransformer(remainder = 'drop',
                       transformers = [
                           ('select', 'passthrough', columns)])

#creating the model
model_1 = Pipeline([
    ('selector', ct),
    ('predictor', DecisionTreeClassifier()),

])


# In[39]:


#fitting the model
model_1.fit(train_df, y);


# In[40]:


#making sure the test data and the train data have same columns
test_correct_columns = pd.DataFrame(test_df, columns=train_df.columns)


# In[41]:


# custom fuction to make submissions
def make_submission(model, test_correct_columns):
    y_test_pred = model.predict(test_correct_columns)
    
    #predictions to dataframe
    predictions = pd.Series(data = y_test_pred,
                           index = test_df.index,
                           name = 'Survived')
    date = pd.Timestamp.now().strftime(format='%Y-%m-%d_%H-%M_')
    predictions.to_csv(f'submission//{date}submission.csv', 
                       index=True, header=True)
    


# In[28]:


#submission to local machine
make_submission(model_1, test_correct_columns)


# In[21]:


#Evaluating the model performance
model_1.score(train_df, y)


# # 0.68 on kaggle(68%)

# In[ ]:




