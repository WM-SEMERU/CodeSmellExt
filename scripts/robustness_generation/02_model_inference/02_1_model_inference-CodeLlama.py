# %% [markdown]
# # Dataset preparation for code-completion

# %%
def default_params(): 
    return {
        'current_model': 'M1',
        'gpu': True,
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'path': '/workspaces/CodeSmells/semeru-datasets/code_smells/generation/curated_500.json',
            'prompt_column': 'prompt',
            'content_column' : 'code',
            'sampling_size': 500,
            'prompt_text' : """complete the following incomplete Python function:\n"""
        },
        'completion_extra_limit': 100,
        'output_generation_dir': '/workspaces/CodeSmells/datax/code_smells/generation/dataset',
        'decoding_strategies' : ['greedy_search', 'beam_search', 'sampling', 'contrastive_search', 'top_k_sampling', 'top_p_sampling'],
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
# ### Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from collections import Counter
import plotly.express as px
from transformers import AutoTokenizer
import torch
import os
import gc

# %%
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from datasets import load_dataset

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

# %%
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %% [markdown]
# ### Model loading

# %%
def instantiate_llm(model_name:str, cache_dir:str):
     '''Instantiate AutoModelForCausalLM'''
     tokenizer = CodeLlamaTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
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

     return tokenizer, model

# %%
tokenizer, pretrained_model = instantiate_llm(params['causal_models'][params['current_model']], params['cache_dir'])

# %%
pretrained_model.config

# %%
pretrained_model.to(device) #WARNING, Verify the device before assigning to memory

# %% [markdown]
# ### Load dataset

# %%
completion_df = pd.read_json(params['dataset']['path'])

# %% [markdown]
# ### Complete prompts

# %%
def generate_text(prompt, decoding_strategy: str, max_length=None):
    # Tokenize the input prompt
    # Tokenize the input prompt with padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Choose the decoding strategy
    if decoding_strategy == "greedy_search":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    elif decoding_strategy == "beam_search":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id)
    elif decoding_strategy == "sampling":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=True, top_k=50, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    elif decoding_strategy == "contrastive_search":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, penalty_alpha=0.6, top_k=4, pad_token_id=tokenizer.pad_token_id)
    elif decoding_strategy == "top_k_sampling":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=True, top_k=50, pad_token_id=tokenizer.pad_token_id)
    elif decoding_strategy == "top_p_sampling":
        output_ids = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=True, top_p=0.92, pad_token_id=tokenizer.pad_token_id)
    else:
        raise ValueError(f"Unsupported decoding strategy: {decoding_strategy}")
    

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"text generated using {decoding_strategy}")

    return generated_text

# %%
def complete_prompts(dataframe):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos_token if not set
    prompt_lenght = len(tokenizer.encode(params['dataset']['prompt_text'], add_special_tokens=False))
    updated_dataframe = dataframe.copy()
    for decoding_strategy in params['decoding_strategies']:
        print(f"======================================== STARTING GENERATION FOR {decoding_strategy} =================================================")
        updated_dataframe[decoding_strategy] = updated_dataframe.apply(lambda row: generate_text(row[params['dataset']['prompt_column']], decoding_strategy ,len(tokenizer.encode(row[params['dataset']['content_column']], add_special_tokens=True)) + prompt_lenght+ params['completion_extra_limit']), axis=1)
        updated_dataframe[decoding_strategy] = updated_dataframe[decoding_strategy].map(lambda completed_code: completed_code[len(params['dataset']['prompt_text']):])
        print(f"======================================== FINISHED GENERATION FOR {decoding_strategy} =================================================")
        output_generation_dir = f"{params['output_generation_dir']}/{params['current_model']}_q_{params['quantization']}/checkpoints"
        create_folder(output_generation_dir)
        updated_dataframe.to_json(f"{output_generation_dir}/curated_generation_{decoding_strategy}_{params['dataset']['sampling_size']}.json")
    return updated_dataframe

# %%
completed_df = complete_prompts(completion_df)

# %% [markdown]
# ### Store dataset 

# %%
output_generation_dir = f"{params['output_generation_dir']}/{params['current_model']}_q_{params['quantization']}"
create_folder(output_generation_dir)
completed_df.to_json(f"{output_generation_dir}/curated_generation_{params['dataset']['sampling_size']}.json")

# %% [markdown]
# ### Clean

# %%
del pretrained_model
torch.cuda.empty_cache()
gc.collect()


