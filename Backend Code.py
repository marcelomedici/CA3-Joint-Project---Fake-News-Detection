#update done on 19.05
#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import seaborn as sns
import os


# In[113]:

import pandas as pd

folder_path = './Untitled Folder/'

df_test = pd.read_csv(folder_path + 'test.tsv', sep='\t', header=None, encoding='utf-8')
df_train = pd.read_csv(folder_path + 'train.tsv', sep='\t', header=None, encoding='utf-8')
df_valid = pd.read_csv(folder_path + 'valid.tsv', sep='\t', header=None, encoding='utf-8')

# In[114]:


columns=['ID','Label','Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts','Context']

# In[115]:


df_test.columns = columns
df_train.columns = columns
df_valid.columns = columns


# In[116]:


df_LIAR = pd.concat([df_test, df_train, df_valid], ignore_index=True)


# In[117]:


Ds_Liar = df_LIAR


# In[118]:


Ds_Liar.describe()


# In[119]:


df_LIAR = pd.concat([df_test, df_train, df_valid], ignore_index=True)


# In[120]:


Ds_Liar = df_LIAR


# In[121]:


Ds_Liar.describe()


# In[122]:


Ds_Liar.isnull().sum()


# In[123]:


Ds_Liar_2 = Ds_Liar.copy()


# In[124]:


Ds_Liar_strnull=['Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

Ds_Liar_Intnull=['Barely_True_Counts','False_Counts','Half_True_Counts','Mostly_True_Counts','Pants_on_fire_Counts']


# In[125]:


Ds_Liar_2[Ds_Liar_strnull] = Ds_Liar_2[Ds_Liar_strnull].fillna('no comments')


# In[126]:


Ds_Liar_2[Ds_Liar_Intnull] = Ds_Liar[Ds_Liar_Intnull].fillna(0)


# In[127]:


Ds_Liar_2.info()


# In[128]:


Ds_Liar_3 = Ds_Liar_2.astype({'ID':'str','Label':'str','Statement':'str','Subject':'str','Speaker':'str','Speakers_Job_Title':'str','State_Info':'str','Affiliation':'str','Barely_True_Counts':'int','False_Counts':'int','Half_True_Counts':'int','Mostly_True_Counts':'int','Pants_on_fire_Counts':'int','Context':'str'})


# In[129]:


Ds_Liar_3 = Ds_Liar_3.rename(columns={'Pants_on_fire_Counts':'deliberately false count','False_Counts':'false counts','Barely_True_Counts':'mostly false count','Half_True_Counts':'partially true count','Mostly_True_Counts':'mostly true count'})


# In[130]:


true_stm = Ds_Liar_3['Label'].value_counts().get('true',0)
Montly_True_stm = Ds_Liar_3['Label'].value_counts().get('mostly-true',0)
Half_True_stm = Ds_Liar_3['Label'].value_counts().get('half-true',0)
Barely_True_stm = Ds_Liar_3['Label'].value_counts().get('barely-true',0)
False_stm = Ds_Liar_3['Label'].value_counts().get('false',0)
Pants_Fire_stm = Ds_Liar_3['Label'].value_counts().get('pants-fire',0)


# In[131]:

print('True', true_stm)
print('Mostly True', Montly_True_stm)
print('Half True', Half_True_stm)
print('Barely True', Barely_True_stm)
print('False', False_stm)
print('Pants on Fire', Pants_Fire_stm)

# In[132]:

Ds_Liar_strnormalization = ['Label','Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']


# In[133]:


for columns in Ds_Liar_strnormalization:
    Ds_Liar_3[columns] = Ds_Liar_3[columns].apply(lambda x: x.lower() if isinstance (x, str) else x)  


# In[134]:


Ds_Liar_3['ID'] = Ds_Liar_3['ID'].str.replace('.json', '',regex=False).astype(int)


# In[135]:


import string

Ds_Liar_str_ponct = ['Statement','Subject','Speaker','Speakers_Job_Title','State_Info','Affiliation','Context']

def ponctual_remove (Ds_Liar_str_ponct):
    if isinstance (Ds_Liar_str_ponct, str):
        return Ds_Liar_str_ponct.translate(str.maketrans('','',string.punctuation))
    return Ds_Liar_str_ponct

for colu in Ds_Liar_str_ponct:
    Ds_Liar_3[colu] =  Ds_Liar_3[colu].apply(ponctual_remove)


# In[136]:


Ds_Liar_3["Label"] = Ds_Liar_3["Label"].str.strip().str.lower()

Ds_Liar_3["Label"] = Ds_Liar_3["Label"].apply(lambda x: "true" if x == "true" else "false")

Ds_Liar_3["labels"] = Ds_Liar_3["Label"].map({"false": 0, "true": 1})

# In[137]:


print(Ds_Liar_3["Label"].value_counts())


# In[138]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Ds_Liar_3[['mostly false count','false counts','partially true count','mostly true count','deliberately false count']]= scaler.fit_transform(Ds_Liar_3[['mostly false count','false counts','partially true count','mostly true count','deliberately false count']])


# In[139]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Ds_Liar_3['Label'] = label_encoder.fit_transform(Ds_Liar_3['Label'])

print(Ds_Liar_3.head())


# In[140]:


!pip install imbalanced-learn


# In[141]:


from imblearn.over_sampling import RandomOverSampler
import pandas as pd

print("original distribution:")
print(Ds_Liar_3["labels"].value_counts())

X = Ds_Liar_3.drop(columns=["labels"])  
y = Ds_Liar_3["labels"]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

Ds_Liar_balanced = X_resampled.copy()
Ds_Liar_balanced["labels"] = y_resampled

print("\ndistribution after oversampling:")
print(Ds_Liar_balanced["labels"].value_counts())




# In[142]:


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

Ds_Liar_balanced.columns = Ds_Liar_balanced.columns.str.strip()
Ds_Liar_hf = Dataset.from_pandas(Ds_Liar_balanced[['Statement', 'labels', 'Context', 'Speakers_Job_Title', 'Subject', 'State_Info', 'Affiliation']])

def tokenization_func(txt):
    statment_tk = tokenizer(txt['Statement'], truncation=True, padding='max_length', max_length=256)
    meta_txt_feature = f"{txt['Context']} [SEP] {txt['Speakers_Job_Title']} [SEP] {txt['Subject']} [SEP] {txt['State_Info']} [SEP] {txt['Affiliation']}"
    meta_concat = tokenizer(meta_txt_feature, padding='max_length', truncation=True, max_length=64)
    input_ids = statment_tk['input_ids'][:-1] + meta_concat['input_ids'][1:]
    attention_mask = statment_tk['attention_mask'][:-1] + meta_concat['attention_mask'][1:]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': txt['labels']
    }

tokenized_df = Ds_Liar_hf.map(tokenization_func, batched=False)

train_dataset, test_dataset = tokenized_df.train_test_split(test_size=0.2, seed=42).values()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    dataloader_num_workers=4,
    fp16=True,
   
    save_total_limit=1,

    save_strategy="no",
    logging_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("finetuned_bert_model")
tokenizer.save_pretrained("finetuned_bert_model")



# In[143]:

!tar -czvf model.tar.gz -C finetuned_bert_model .

import sagemaker
import boto3

session = sagemaker.Session()
bucket = session.default_bucket() 
prefix = "bert-model"
model_s3_path = f"s3://{bucket}/{prefix}/model.tar.gz"

s3 = boto3.client('s3')
s3.upload_file('model.tar.gz', bucket, f'{prefix}/model.tar.gz')
print("Modelo enviado para:", model_s3_path)

from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data=model_s3_path,
    role=sagemaker.get_execution_role(),
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    env={
        'HF_TASK': 'text-classification'
    }
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

response = predictor.predict({
    "inputs": "statement Check!"
})

print("answer:", response)



# In[144]:


import sagemaker

role = sagemaker.get_execution_role()
print(role)


# In[145]:


from sagemaker.huggingface.model import HuggingFaceModel

role = 'arn:aws:iam::411097365479:role/service-role/AmazonSageMaker-ExecutionRole-20250517T180716'
model_data = 's3://sagemaker-us-east-2-411097365479/bert-model/model.tar.gz'

huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

print("Endpoint:", predictor.endpoint_name)


