#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv(r'C:\Users\Fidha\Desktop\Sem3 datasets\india-news-headlines.csv')
data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe()


# ### DATA PRE-PROCESSING

# In[9]:


data.isnull()


# In[10]:


data.isnull().sum()


# In[11]:


data.duplicated()


# In[12]:


data.duplicated().sum()


# In[13]:


data.drop_duplicates(inplace=True)


# In[14]:


data.duplicated().sum()


# ### TD-IDF

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


# tfidf calculation
text_content = data['headline_text']
vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                             #min_df=8,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
tfidf = vector.fit_transform(text_content)


# In[32]:


data = pd.DataFrame(tfidf[0].T.todense(), index = vector.get_feature_names(), columns=["TF-IDF"])

data = df.sort_values('TF-IDF', ascending=False)


# In[33]:


data


# In[ ]:


# Distributed,quo, vajpajee, ayodhya are the most important words in this dataset

