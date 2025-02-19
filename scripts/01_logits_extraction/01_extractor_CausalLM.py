# %% [markdown]
# # Logits Extractor Module

# %%
# %%
# Transformation names: 'RenameVariable-1' 'RenameVariable-2'
#                       'Add2Equal' 'SwitchEqualExp''InfixDividing' 
#                       'SwitchRelation' 
def default_params(): 
    return {
        'current_model': 'M1',
        'gpu': True,
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'path': '/workspaces/CodeSmells/semeru-datasets/code_smells/extraction',
            'transformation': 'SwitchEqualExp',
            'content_column': 'code',
            'sampling_size': 500,
        },
        'logging_path': '/workspaces/CodeSmells/datax/code_smells/logs', 
        'callbacks_dir' : '/workspaces/CodeSmells/datax/code_smells/callbacks',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'causal_models': {
            'M1' : 'codellama/CodeLlama-7b-hf', #https://huggingface.co/codellama/CodeLlama-7b-hf, 
            'M2' : 'mistralai/Mistral-7B-v0.3', #https://huggingface.co/mistralai/Mistral-7B-v0.3,
            'M3' : 'microsoft/Phi-3.5-mini-instruct', #https://huggingface.co/microsoft/Phi-3.5-mini-instruct 
            'M4' : 'Qwen/Qwen2.5-Coder-7B', #https://huggingface.co/Qwen/Qwen2.5-Coder-7B
            'M5' : 'facebook/incoder-6B', #https://huggingface.co/facebook/incoder-6B
            'M6' : 'bigcode/starcoder2-7b', #https://huggingface.co/bigcode/starcoder2-7b 
            'M7' : 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', #https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
            'M8' : 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', #https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
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
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%
# Define log file path
log_file = f"{params['logging_path']}/{params['current_model']}/{params['dataset']['transformation']}"
create_folder(log_file)
log_file += '/log.txt'

# Create the log file if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, 'w'): 
        pass  # Create an empty log file

# %%
import logging
logging.basicConfig(filename=log_file, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %% [markdown]
# #### GPU

# %%
#! nvidia-smi

# %%
torch.__version__

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() and params['gpu'] else "cpu")
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
df_dataset = pd.read_json(f"{params['dataset']['path']}/{params['dataset']['transformation']}_{params['dataset']['sampling_size']}.json", )

# %%
#df_dataset = df_dataset[df_dataset['input_lenght']>=700]
#df_dataset = df_dataset[:20]

# %%
df_dataset.head(5)

# %% [markdown]
# #### Logit Inference

# %%
def logit_extractor(model, batch, tf_encoded_inputs, from_index=0):
    """
    Output is the class CausalLMOutputWithPast (https://huggingface.co/transformers/v4.10.1/main_classes/output.html?highlight=causallmoutputwithpast)"
    logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) – Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    The expression i.type(torch.LongTensor).to(device) is for casting labels for the loss
    """
    callbacks_dir = f"{params['callbacks_dir']}/{params['current_model']}_q_{params['quantization']}/{params['dataset']['transformation']}"
    create_folder(callbacks_dir)
    
    for idx, n in enumerate(range(from_index, len(tf_encoded_inputs), batch)):
        torch.cuda.empty_cache()
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
            np.save(f"{callbacks_dir}/logits_tensor[{jdx+n}]_batch[{current_batch}].npy", o_logits)
        np.save(f"{callbacks_dir}/_loss_batch[{current_batch}].npy", output_loss)
        
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

print("================================= PROCESS COMPLETE =================================")

# %%
torch.cuda.empty_cache()
gc.collect()
del model


