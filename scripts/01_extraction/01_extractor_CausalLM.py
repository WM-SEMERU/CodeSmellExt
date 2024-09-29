# %% [markdown]
# # Logits Extractor Module

# %%
def default_params(): 
    return {
        'current_model': 'M2', 
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'name': '/workspaces/CodeSmells/semeru-datasets/code_smells/codesmell_dataset.csv',
            'content_column': 'code', 
            'number_samples': 55,
        },
        'default_max_position_embeddings' : 16384,
        'output_path': '../data/raw_logits',
        'preprocessed_dataset_dir' : '../datax/code_smells/dataset_preprocessing',
        'cache_dir': '../datax/hugging_face_cache',
        'log_file': '../datax/code_smells/logit_extraction.log', 
        'callbacks_dir' : '../datax/code_smells/callbacks',
        'causal_models': {
            'M1': 'codellama/CodeLlama-7b-hf', #https://huggingface.co/codellama/CodeLlama-7b-hf, 
            'M2': 'mistralai/Mistral-7B-v0.3', #https://huggingface.co/mistralai/Mistral-7B-v0.3
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
import seaborn as sns
from scipy import stats
from statistics import NormalDist
import matplotlib.pyplot as plt

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# %%
import logging
logging.basicConfig(filename=params['log_file'], format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %% [markdown]
# #### GPU

# %%
! nvidia-smi

# %%
torch.__version__

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %%
torch.cuda.memory_allocated()

# %% [markdown]
# ## Logits Extractor
# >
# > Extracting Tensor Logits from a given Neural Code Model
# >

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

# %%
model.config

# %%
model.to(device) #WARNING, Verify the device before assigning to memory

# %% [markdown]
# #### Dataset

# %%
df_dataset = pd.read_json(params['preprocessed_dataset_dir'] + '/' + params['current_model'] + '_q_' + params['quantization'] +'.json', )

# %% [markdown]
# #### Logit Inference

# %%
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%
def logit_extractor(model, batch, tf_encoded_inputs, from_index=0):
    """
    Output is the class CausalLMOutputWithPast (https://huggingface.co/transformers/v4.10.1/main_classes/output.html?highlight=causallmoutputwithpast)"
    logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) – Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    The expression i.type(torch.LongTensor).to(device) is for casting labels for the loss
    """
    create_folder(params['callbacks_dir']+ '/'+ params['current_model'] + '_q_' + params['quantization'])
    
    for idx, n in enumerate(range(from_index, len(tf_encoded_inputs), batch)):
        output = []
        for encoded_sample in tf_encoded_inputs[n:n+batch]:
            output.append( 
                model(input_ids = encoded_sample, labels = encoded_sample)
            )
        output_logits = [ o['logits'].detach().to('cpu').numpy() for o in output ]  #Logits Extraction
        output_loss = np.array([ o.loss.detach().to('cpu').numpy() for o in output ])  #Language modeling loss (for next-token prediction).

        #Saving Callbacks
        current_batch = idx + (from_index//batch)
        for jdx, o_logits in enumerate(output_logits):
            np.save( params['callbacks_dir']+ '/'+ params['current_model'] + '_q_' + params['quantization'] + '/' + f'logits_tensor[{jdx+n}]_batch[{current_batch}].npy', o_logits)
        np.save( params['callbacks_dir']+ '/'+ params['current_model'] + '_q_' + params['quantization'] + '/' + f'_loss_batch[{current_batch}].npy', output_loss)

        print(f"Batch [{current_batch}] Completed")

        #Memory Released
        for out in output:
            del out.logits
            torch.cuda.empty_cache()
            del out.loss
            torch.cuda.empty_cache()
        for out in output_logits:
            del out
            torch.cuda.empty_cache()
        for out in output_loss:
            del out
            torch.cuda.empty_cache()

# %%
#Casting Integers to Tensor Integers. Make sure the tesor is created in a device
#We ignored the parameter attention_mask since we are not using masking here [https://huggingface.co/transformers/v4.10.1/glossary.html#attention-mask]
tf_encoded_inputs = [tokenizer(sample, return_tensors='pt')['input_ids'].to(device) for sample in df_dataset[params['dataset']['content_column']].values]

# %%
## ACTUAL EXPERIMENT
## TIME AND MEMORY CONSUMING
logit_extractor(
    model = model,
    batch = 1, 
    tf_encoded_inputs = tf_encoded_inputs, 
    from_index=0
)

# %%
torch.cuda.empty_cache()
gc.collect()


