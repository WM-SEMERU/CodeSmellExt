# %% [markdown]
# # Logits Preprocessing and Data Engineering

# %%
def default_params(): 
    return {
        'current_model': 'M1', 
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'name': '/workspaces/CodeSmells/semeru-datasets/code_smells/codesmell_dataset.csv',
            'content_column': 'code', 
        },
        'default_max_position_embeddings' : 16384,
        'output_path': '/workspaces/CodeSmells/data/raw_logits',
        'preprocessed_dataset_dir' : '/workspaces/CodeSmells/datax/code_smells/dataset_preprocessing',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'log_file': '/workspaces/CodeSmells/scripts/02_data_engineering/logit_extraction.log', 
        'callbacks_dir' : '/workspaces/CodeSmells/datax/code_smells/callbacks',
        'causal_models': {
            'M1': 'codellama/CodeLlama-7b-hf', #https://huggingface.co/codellama/CodeLlama-7b-hf
        },
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

# %%
from transformers import CodeLlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

# %%
import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(
    filename=params['log_file'],
    filemode='a',
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO
    )

# %%
import seaborn as sns
from scipy import stats
from statistics import NormalDist
import matplotlib.pyplot as plt

# %% [markdown]
# #### Dataset

# %%
df_dataset = pd.read_json(params['preprocessed_dataset_dir'] + '/' + params['current_model'] + '_q_' + params['quantization'] + '.json',)

# %%
df_dataset.reset_index(drop=True, inplace=True)

# %% [markdown]
# #### Model Loading

# %%
def instantiate_llm(model_name:str, cache_dir:str):
     '''Instantiate AutoModelForCausalLM'''
     tokenizer = CodeLlamaTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
     logging.info("Loaded AutoTokenizer - " + model_name)
     model = None
     if params['quantization'] == 'int4':
          model = LlamaForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, load_in_4bit=True)
     elif params['quantization'] == 'int8':
          model = LlamaForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, load_in_8bit=True)
     elif params['quantization'] == 'float32':
          model = LlamaForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float32)
     elif params['quantization'] == 'float16':
          model = LlamaForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float16)
     else: 
          model = LlamaForCausalLM.from_pretrained(model_name, cache_dir = cache_dir)
     logging.info("Loaded AutoModelForCausalLM - " + model_name)

     return tokenizer, model

# %%
tokenizer, model = instantiate_llm(params['causal_models'][params['current_model']], params['cache_dir'])

# %% [markdown]
# #### Softmax Normalization and Data Engineering

# %%
def topk_tuple( logit_vocab_tensor, largest, tokenizer_fn):
    "Run topk for a token"
    topk = logit_vocab_tensor.topk( k=1 , largest=largest ) #TODO K number of elements can be extended
    return ( tokenizer.convert_tokens_to_string([tokenizer_fn.decode(topk.indices)]), topk.values.item())

def min_max_logits( logit_vocab_sample_tensor, tokenizer_fn ):
    "Compute min_max for a sample"
    max_cases = []
    min_cases = []
    for logit_vocab_tensor in logit_vocab_sample_tensor:
        max_cases.append( topk_tuple( logit_vocab_tensor = logit_vocab_tensor, largest = True, tokenizer_fn = tokenizer_fn) ) #TST Max Logit
        min_cases.append( topk_tuple( logit_vocab_tensor = logit_vocab_tensor, largest = False, tokenizer_fn = tokenizer_fn) ) #TST Min Logit
    return max_cases, min_cases

def actual_logit( 
                 logit_vocab_sample_tensor, 
                 tokenized_prompt, 
                 tokenizer_fn,
                 ):
    "Compute actual logits for a sample"
    actual_logits_prompt = []
    for token_pos, id_token in enumerate( tokenized_prompt[1:] ): #Eliminate the first token prediction since we do not use it
        actual_logits_prompt.append(
            (   tokenizer.convert_tokens_to_string([tokenizer_fn.decode( int(id_token))]), #retrieving the name of the token with the id
                logit_vocab_sample_tensor[token_pos][int(id_token)].item()) #retrieving the logit given the position in the sequence and the position in the vocab
            )
    return actual_logits_prompt

# %%
soft = torch.nn.Softmax( dim = 0 ) #Flattening normalization

# %%
out= np.load(params['callbacks_dir']+ '/'+ params['current_model'] + '_q_' + params['quantization'] +'/' + 'logits_tensor[0]_batch[0].npy')
print(out.shape) #<sample,tokens,voc_tokens>
out = out[0]


# %%
max_case,min_case = min_max_logits(
    logit_vocab_sample_tensor = [ soft( torch.from_numpy(token) ) for token in out], ####### 
    tokenizer_fn= tokenizer
    )
print(max_case)
assert len(max_case) == len(min_case)

# %%
assert tokenizer.decode(df_dataset['input_ids'][0]) == df_dataset[params['dataset']['content_column']][0]
df_dataset[params['dataset']['content_column']][0]

# %%
input_ids_list = tokenizer.batch_encode_plus(df_dataset[params['dataset']['content_column']].tolist())
input_ids_list = [torch.tensor(  input_ids, dtype = torch.int) for input_ids in input_ids_list.input_ids]

actual_cases = actual_logit(
    logit_vocab_sample_tensor = [ soft( torch.from_numpy(token) ) for token in out] , #Out is a complete sequence
    tokenized_prompt = input_ids_list[0], ## SAMPLE ID
    tokenizer_fn = tokenizer
    )
actual_cases

# %% [markdown]
# #### Processing all the Batches

# %%
def batching_logits(tokenizer,tf_input_ids,size=10000):
    max_logit_token_prompt = []
    min_logit_token_prompt = []
    actual_logit_token_prompt = []

    
    soft = torch.nn.Softmax( dim = 0 )                          #Flattening normalization
    
    for file in range( size ):
        out = np.load(params['callbacks_dir']+ '/'+ params['current_model'] + '_q_' + params['quantization'] +'/'+ f'logits_tensor[{file}]_batch[{file}].npy') #<sample,tokens,voc_tokens>
        out = out[0]  ##### #<tokens,voc_tokens>
        next_tokens_distribution = [ soft( torch.from_numpy(token) ) for token in out]  #Flattening normalization
        
        max_cases,min_cases = min_max_logits(
            logit_vocab_sample_tensor = next_tokens_distribution,
            tokenizer_fn= tokenizer
            )

        actual_cases = actual_logit(
            logit_vocab_sample_tensor = next_tokens_distribution,
            tokenized_prompt = tf_input_ids[ file ],
            tokenizer_fn = tokenizer
            )
        
        max_logit_token_prompt.append( max_cases )
        min_logit_token_prompt.append( min_cases )
        actual_logit_token_prompt.append( actual_cases )
        
        logging.info(file)
    return max_logit_token_prompt,min_logit_token_prompt,actual_logit_token_prompt

# %%
input_ids_list = tokenizer.batch_encode_plus(df_dataset[params['dataset']['content_column']].tolist())
input_ids_list = [torch.tensor(  input_ids, dtype = torch.int) for input_ids in input_ids_list.input_ids]

# %%
max_logit_token_prompt, min_logit_token_prompt, actual_logit_token_prompt = batching_logits(
    tokenizer=tokenizer , tf_input_ids=input_ids_list, 
    size = len(df_dataset)
) #<---WARNING TIME Consuming

# %% [markdown]
# #### Saving results

# %%
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%
dataframe_to_save = df_dataset.copy()
dataframe_to_save['max_prob'] = max_logit_token_prompt
dataframe_to_save['min_prob'] = min_logit_token_prompt
dataframe_to_save['actual_prob'] = actual_logit_token_prompt
dataframe_to_save.shape

# %%
dataframe_to_save.head(5)

# %%
create_folder(params['output_path'] + '/' + params['current_model'] + '_q_' + params['quantization'])
dataframe_to_save.to_csv( params['output_path'] + '/' + params['current_model'] + '_q_' + params['quantization'] + '/' + 'raw_logits.csv')

# %% [markdown]
# #### Loss Retrieval

# %%
def batching_loss( size = dataframe_to_save.shape[0] ):
    output_loss = []
    for current_batch in range(size):
        out = np.load(params['callbacks_dir']+ '/'+ params['current_model'] +  '_q_' + params['quantization'] +'/' + f'_loss_batch[{current_batch}].npy') 
        output_loss.append( out.item() ) #.item() for numpy library
        logging.info(current_batch)
    return output_loss

# %%
output_loss = batching_loss() #[WAENING!] Takes Time

# %%
output_loss

# %%
dataframe_to_save['loss'] = output_loss
dataframe_to_save.head(5)

# %%
## Saving CheckPoint 2
dataframe_to_save.to_csv( params['output_path'] + '/' + params['current_model'] + '_q_' + params['quantization'] + '/' + 'raw_logits.csv')

# %%
torch.cuda.empty_cache()
gc.collect()