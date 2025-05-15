#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[113]:


os.chdir(r'C:\Users\medic\Desktop\College\Joint Project\Dataset\LIAR dataset')

df_test = pd.read_csv('test.tsv', sep='\t', header=None)

df_train = pd.read_csv('train.tsv', sep='\t', header=None)

df_valid = pd.read_csv('valid.tsv', sep='\t', header=None)


# In[114]:


columns=['ID','Label','Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts','Context']


# In[115]:


df_test.columns = columns
df_train.columns = columns
df_valid.columns = columns


# In[116]:


df_test.head()


# In[117]:


df_train.head()


# In[118]:


df_valid.head()


# In[119]:


df_LIAR = pd.concat([df_test, df_train, df_valid], ignore_index=True)


# In[120]:


df_LIAR.head()


# In[121]:


os.chdir(r'C:\Users\medic\Desktop\College\Joint Project\Dataset\LIAR dataset')

Ds_Liar = pd.read_csv('LIAR.CSV')


# In[122]:


Ds_Liar.head()


# In[123]:


Ds_Liar.describe()


# In[124]:


Ds_Liar.info()


# In[125]:


Ds_Liar.isnull().sum()


# In[126]:


Ds_Liar_2 = Ds_Liar.copy()


# In[127]:


Ds_Liar_strnull=['Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

Ds_Liar_Intnull=['Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts']


# In[128]:


Ds_Liar_2[Ds_Liar_strnull] = Ds_Liar_2[Ds_Liar_strnull].fillna('no comments')


# In[129]:


Ds_Liar_2[Ds_Liar_Intnull] = Ds_Liar[Ds_Liar_Intnull].fillna(0)


# In[130]:


Ds_Liar_2.info()


# In[131]:


Ds_Liar_3 = Ds_Liar_2.astype({'ID':'str','Label':'str','Statement':'str','Subject':'str','Speaker':'str','Speakers_Job_Title':'str','State_Info':'str','Affiliation':'str','Barely_True_Counts':'int','False_Counts':'int','Half_True_Counts':'int','Mostly_True_Counts':'int','Pants_on_fire_Counts':'int','Context':'str'})


# In[132]:


Ds_Liar_3 = Ds_Liar_3.rename(columns={'Pants_on_fire_Counts':'deliberately false count','False_Counts':'false counts','Barely_True_Counts':'mostly false count','Half_True_Counts':'partially true count','Mostly_True_Counts':'mostly true count'})


# In[133]:


Ds_Liar_3['Label']= Ds_Liar_3['Label'].replace({'pants-fire':'deliberately false','false':'false','barely-true':'mostly false','half-true':'partially true','mostly-true':'mostly true','true':'true'})


# In[134]:


Ds_Liar_3


# In[135]:


Ds_Liar_3.info()


# In[136]:


true_stm = Ds_Liar_3['Label'].value_counts().get('true',0)
Montly_True_stm = Ds_Liar_3['Label'].value_counts().get('mostly-true',0)
Half_True_stm = Ds_Liar_3['Label'].value_counts().get('half-true',0)
Barely_True_stm = Ds_Liar_3['Label'].value_counts().get('barely-true',0)
False_stm = Ds_Liar_3['Label'].value_counts().get('false',0)
Pants_Fire_stm = Ds_Liar_3['Label'].value_counts().get('pants-fire',0)


# In[137]:


print('True', true_stm)
print('Mostly True', Montly_True_stm)
print('Half True', Half_True_stm)
print('Barely True', Barely_True_stm)
print('False', False_stm)
print('Pants on Fire', Pants_Fire_stm)


# In[138]:


#Normalization - Starting process


# In[139]:


Ds_Liar_strnormalization = ['Label','Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']


# In[140]:


for columns in Ds_Liar_strnormalization:
    Ds_Liar_3[columns] = Ds_Liar_3[columns].apply(lambda x: x.lower() if isinstance (x, str) else x)  


# In[141]:


Ds_Liar_3


# In[142]:


Ds_Liar_3['ID'] = Ds_Liar_3['ID'].str.replace('.json', '',regex=False).astype(int)


# In[143]:


Ds_Liar_3


# In[144]:


#Pre-processing ponctuation removal
import string

Ds_Liar_str_ponct = ['Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

def ponctual_remove (Ds_Liar_str_ponct):
    if isinstance (Ds_Liar_str_ponct, str):
        return Ds_Liar_str_ponct.translate(str.maketrans('','',string.punctuation))
    return Ds_Liar_str_ponct

for colu in Ds_Liar_str_ponct:
    Ds_Liar_3[colu] =  Ds_Liar_3[colu].apply(ponctual_remove)


# In[145]:


Ds_Liar_3


# In[146]:


get_ipython().system('pip install torch')


# In[147]:


from transformers import BertTokenizer, BertModel
import torch

TokenBert = BertTokenizer.from_pretrained('bert-base-uncased')
ModelBert = BertModel.from_pretrained('bert-base-uncased')

def vector_cls (txt):
    tokens_Bert = TokenBert(txt, return_tensors ='pt', truncation = True, max_length = 512)
    with torch.no_grad():
        outputs = ModelBert(**tokens_Bert)
    
    embeddings_Bert = outputs.last_hidden_state
    cls_vector = embeddings_Bert[0][0]
    return cls_vector.numpy()
    
collumns_emb = ['Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

for col in collumns_emb:
    Ds_Liar_3[f'{col}_cls_embedding'] = Ds_Liar_3[col].apply(vector_cls)

print(Ds_Liar_3.head())

Ds_Liar_3.to_csv('df_embeddings.csv', index=False)


# In[156]:


os.chdir(r'C:\Users\medic\Desktop\College\Joint Project\Dataset\LIAR dataset')
df_embeddings = pd.read_csv('df_embeddings.csv')
df_embeddings


# In[180]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_embeddings[['mostly false count','false counts','partially true count','mostly true count','deliberately false count']]= scaler.fit_transform(df_embeddings[['mostly false count','false counts','partially true count','mostly true count','deliberately false count']])


# In[181]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df_embeddings['Label'] = label_encoder.fit_transform(df_embeddings['Label'])

print(df_embeddings.head())


# In[186]:


delete_columns = ['Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']  # Lista com os nomes das colunas

# Deletando várias colunas
df_embeddings = df_embeddings.drop(delete_columns, axis=1)


# In[185]:


deleling_ID = ['ID'] 

df_embeddings = df_embeddings.drop(deleling_ID, axis=1)

print(df_embeddings.head())


# In[187]:


df_embeddings.to_csv('embedded_liar.csv', index=False)


# In[212]:


df_embeddings


# In[210]:


def enbeddings_to_float(enbedding_obj):
    if isinstance(enbedding_obj, list):
        return enbedding_obj
    
    if isinstance(enbedding_obj, str):
        enbedding_features = enbedding_obj.strip('[]').split(',')
        enbedding_features = [float(x) for x in enbedding_features if x.strip()]
        return enbedding_features

embedding_col = [
    'Statement_cls_embedding', 
    'Subject_cls_embedding', 
    'Speaker_cls_embedding', 
    'Speakers_Job_Title_cls_embedding', 
    'State_Info_cls_embedding', 
    'Affiliation_cls_embedding', 
    'Context_cls_embedding'
]

for coll in embedding_col:
    df_embeddings[coll] = df_embeddings[coll].apply(enbeddings_to_float)


# In[214]:


print(df_embeddings[embedding_col].applymap(type).head())


# In[211]:


df_embeddings


# In[213]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

X = df_embeddings.drop(columns=['Label'])
y = df_embeddings['Label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc:.4f}")

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {accuracy}")

joblib.dump(model, 'model_classification.pkl')


# In[ ]:





# In[ ]:




