#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


os.chdir(r'C:\Users\medic\Desktop\College\Joint Project\Dataset\LIAR dataset')

df_test = pd.read_csv('test.tsv', sep='\t', header=None)

df_train = pd.read_csv('train.tsv', sep='\t', header=None)

df_valid = pd.read_csv('valid.tsv', sep='\t', header=None)


# In[4]:


columns=['ID','Label','Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts','Context']


# In[5]:


df_test.columns = columns
df_train.columns = columns
df_valid.columns = columns


# In[6]:


df_test.head()


# In[7]:


df_train.head()


# In[8]:


df_valid.head()


# In[9]:


df_LIAR = pd.concat([df_test, df_train, df_valid], ignore_index=True)


# In[10]:


df_LIAR.head()


# In[11]:


df_LIAR.to_csv('LIAR.CSV', index=False)


# In[12]:


os.chdir(r'C:\Users\medic\Desktop\College\Joint Project\Dataset\LIAR dataset')

Ds_Liar = pd.read_csv('LIAR.CSV')


# In[13]:


Ds_Liar.head()


# In[14]:


Ds_Liar.describe()


# In[15]:


Ds_Liar.info()


# In[16]:


Ds_Liar.isnull().sum()


# In[17]:


Ds_Liar_2 = Ds_Liar.copy()


# In[18]:


Ds_Liar_strnull=['Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

Ds_Liar_Intnull=['Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts']


# In[19]:


Ds_Liar_2[Ds_Liar_strnull] = Ds_Liar_2[Ds_Liar_strnull].fillna('no comments')


# In[20]:


Ds_Liar_2[Ds_Liar_Intnull] = Ds_Liar[Ds_Liar_Intnull].fillna(0)


# In[21]:


Ds_Liar_2.info()


# In[22]:


Ds_Liar_3 = Ds_Liar_2.astype({'ID':'str','Label':'str','Statement':'str','Subject':'str','Speaker':'str','Speakers_Job_Title':'str','State_Info':'str','Affiliation':'str','Barely_True_Counts':'int','False_Counts':'int','Half_True_Counts':'int','Mostly_True_Counts':'int','Pants_on_fire_Counts':'int','Context':'str'})


# In[53]:


Ds_Liar_3.info()


# In[66]:


true_stm = Ds_Liar_3['Label'].value_counts().get('true',0)
Montly_True_stm = Ds_Liar_3['Label'].value_counts().get('mostly-true',0)
Half_True_stm = Ds_Liar_3['Label'].value_counts().get('half-true',0)
Barely_True_stm = Ds_Liar_3['Label'].value_counts().get('barely-true',0)
False_stm = Ds_Liar_3['Label'].value_counts().get('false',0)
Pants_Fire_stm = Ds_Liar_3['Label'].value_counts().get('pants-fire',0)


# In[67]:


print('True', true_stm)
print('Mostly True', Montly_True_stm)
print('Half True', Half_True_stm)
print('Barely True', Barely_True_stm)
print('False', False_stm)
print('Pants on Fire', Pants_Fire_stm)


# In[ ]:


#Normalization - Starting process

