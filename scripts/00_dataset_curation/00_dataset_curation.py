# %% [markdown]
# # Dataset Preprocessing

# %%
def default_params(): 
    return {
        'current_model': 'M1', 
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'name': '/workspaces/CodeSmells/semeru-datasets/code_smells/codesmell_dataset.csv',
            'content_column': 'code', 
            'number_samples': 156151,
        },
        'default_max_position_embeddings' : 16384, ### CodeLlama Limit. 
        'preprocessed_dataset_dir': '/workspaces/CodeSmells/datax/code_smells/dataset_preprocessing',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'log_file': '/workspaces/CodeSmells/datax/code_smells/dataset_preprocessing.log', 
        'causal_models': {
            'M1': 'codellama/CodeLlama-7b-hf', #https://huggingface.co/codellama/CodeLlama-7b-hf, 
            'M2': 'mistralai/Mistral-7B-v0.3', #https://huggingface.co/mistralai/Mistral-7B-v0.3
        }
    }
params = default_params()

# %% [markdown]
# #### Imports

# %%
import pandas as pd
import os
import time
import numpy as np
import torch
import gc
import seaborn as sns
from scipy import stats
from statistics import NormalDist
import matplotlib.pyplot as plt

# %%
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

# %%
import logging
logging.basicConfig(filename=params['log_file'], format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %% [markdown]
# #### Model Loading

# %%
def instantiate_llm(model_name:str, cache_dir:str):
     '''Instantiate AutoModelForCausalLM'''
     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
     logging.info("Loaded AutoTokenizer - " + model_name)
     model = None
     if params['quantization'] == 'int4':
          model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, load_in_4bit=True)
     elif params['quantization'] == 'int8':
          model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, load_in_8bit=True)
     elif params['quantization'] == 'float32':
          model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float32)
     elif params['quantization'] == 'float16':
          model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float16)
     else: 
          model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir)
     logging.info("Loaded AutoModelForCausalLM - " + model_name)

     return tokenizer, model

# %%
tokenizer, model = instantiate_llm(params['causal_models'][params['current_model']], params['cache_dir'])

# %% [markdown]
# #### Dataset Curation

# %%
dataset = load_dataset('csv', data_files=params['dataset']['name'], cache_dir=params['cache_dir']) ## HF_DATASET
dataset

# %%
df_dataset = pd.concat([dataset[key].to_pandas() for key in dataset.keys()])
df_dataset = df_dataset[:params['dataset']['number_samples']]

# %%
df_dataset = df_dataset.drop(['Unnamed: 0'], axis=1) ### TODO: FIX unamed column, also add method_id

# %%


# %%
### append input_ids 
df_dataset['input_ids'] = df_dataset[params['dataset']['content_column']].apply(lambda code: tokenizer(code, add_special_tokens=False)['input_ids'])

# %%
max_position_embeddings = params['default_max_position_embeddings']
#max_position_embeddings = model.max_position_embeddings
df_dataset = df_dataset[df_dataset['input_ids'].apply(lambda ids: len(ids) <= max_position_embeddings)]

# %% [markdown]
# #### Save Dataset

# %%
df_dataset.to_json(params['preprocessed_dataset_dir'] + '/' + params['current_model'] + '_q_' + params['quantization'] +'.json')

# %%
torch.cuda.empty_cache()
gc.collect()


