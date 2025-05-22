# %% [markdown]
# # Logits Preprocessing and Data Engineering

# %%
def default_params(): 
    return {
        'current_model': 'S1', 
        'quantization': 'none', #['none',"int4", "int8", "float32", "float16"]
        'dataset': {
            'path': '/workspaces/CodeSmells/semeru-datasets/code_smells/pipeline/curated',
            'content_column': 'code',
            'sampling_size': 500,
        },
        'logging_path': '/workspaces/CodeSmells/datax/code_smells/logs', 
        'output_path' : '/workspaces/CodeSmells/datax/code_smells/logits/pipeline',
        'callbacks_path' : '/workspaces/CodeSmells/datax/code_smells/callbacks/pipeline',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'causal_models': {
            ##### BY ARCHITECTURE, SAME SIZE #####
            'M1' : 'codellama/CodeLlama-7b-hf', #https://huggingface.co/codellama/CodeLlama-7b-hf, 
            'M2' : 'mistralai/Mistral-7B-v0.3', #https://huggingface.co/mistralai/Mistral-7B-v0.3,
            'M3' : 'Qwen/Qwen2.5-Coder-7B', #https://huggingface.co/Qwen/Qwen2.5-Coder-7B,
            'M4' : 'bigcode/starcoder2-7b', #https://huggingface.co/bigcode/starcoder2-7b,
            ##### BY SIZE, SAME ARCHITECTURE #####
            'S1' : 'Qwen/Qwen2.5-Coder-0.5B', #https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B,
            'S2' : 'Qwen/Qwen2.5-Coder-1.5B', #https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B,
            'S3' : 'Qwen/Qwen2.5-Coder-3B', #https://huggingface.co/Qwen/Qwen2.5-Coder-3B,
            'S4' : 'Qwen/Qwen2.5-Coder-7B', #https://huggingface.co/Qwen/Qwen2.5-Coder-7B,
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
log_file = f"{params['logging_path']}/{params['current_model']}"
create_folder(log_file)
log_file += '/data_en.txt'

# Create the log file if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, 'w'): 
        pass  # Create an empty log file

# %%
import logging
logging.basicConfig(filename=log_file, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %% [markdown]
# #### Dataset

# %%
print(f"{params['dataset']['path']}_{params['dataset']['sampling_size']}.json")

# %%
df_dataset = pd.read_json(f"{params['dataset']['path']}_{params['dataset']['sampling_size']}.json", )

# %%
df_dataset

# %% [markdown]
# #### Model Loading

# %%
def instantiate_llm(model_name:str, cache_dir:str):
     '''Instantiate AutoModelForCausalLM'''
     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir, use_fast=True)
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
print(model.config)
print(model.__class__)
print(tokenizer.__class__)

# %% [markdown]
# #### preprocess dataset

# %%
df_dataset['input_ids'] = df_dataset[params['dataset']['content_column']].map(lambda code: tokenizer.encode(code, add_special_tokens=False))
df_dataset['input_lenght'] = df_dataset['input_ids'].map(lambda input_ids: len(input_ids))

# %% [markdown]
# #### Softmax Normalization and Data Engineering

# %%
def topk_tuple(logit_vocab_tensor, largest, tokenizer_fn):
    """
    Return the decoded top-1 (or bottom-1) token and its logit value.
    """
    topk = logit_vocab_tensor.topk(k=1, largest=largest)
    top_token_id = topk.indices[0].item()
    decoded_token = tokenizer_fn.decode([top_token_id])  # Already handles special tokens and spaces
    return (decoded_token, topk.values[0].item())

# %%
def analyze_logits(logit_tensor_sequence, input_token_ids, tokenizer_fn, skip_first_token=True):
    """
    Analyze logits for each token in a sequence.

    Args:
        logit_tensor_sequence (List[Tensor]): List of vocab-sized logits for each token position.
        input_token_ids (List[int] or Tensor): Token IDs of the input prompt.
        tokenizer_fn: HuggingFace tokenizer with .decode() method.
        skip_first_token (bool): Whether to skip the first token prediction (default: True).

    Returns:
        dict with:
            - "max_cases": list of (decoded top-1 token, logit value)
            - "min_cases": list of (decoded bottom-1 token, logit value)
            - "actual_logits": list of (decoded ground-truth token, logit value)
    """
    max_cases = []
    min_cases = []
    actual_logits = []

    start_index = 1 if skip_first_token else 0
    token_targets = input_token_ids[start_index:]

    for position, token_id in enumerate(token_targets):
        vocab_logits = logit_tensor_sequence[position]

        # Top-1 max and min predictions
        max_case = topk_tuple(logit_vocab_tensor=vocab_logits, largest=True, tokenizer_fn=tokenizer_fn)
        min_case = topk_tuple(logit_vocab_tensor=vocab_logits, largest=False, tokenizer_fn=tokenizer_fn)

        # Actual token logit
        decoded_token = tokenizer_fn.decode([int(token_id)])
        logit_value = vocab_logits[int(token_id)].item()
        actual_case = (decoded_token, logit_value)

        max_cases.append(max_case)
        min_cases.append(min_case)
        actual_logits.append(actual_case)

    return {
        "max_cases": max_cases,
        "min_cases": min_cases,
        "actual_logits": actual_logits
    }

# %%
soft = torch.nn.Softmax( dim = 0 ) #Flattening normalization

# %%
callbacks_dir = f"{params['callbacks_path']}/{params['current_model']}_q_{params['quantization']}"
out = np.load(f"{callbacks_dir}/logits_tensor[0]_batch[0].npy")

print(out.shape) #<sample,tokens,voc_tokens>
out = out[0]


# %%
input_ids_list = tokenizer.batch_encode_plus(df_dataset[params['dataset']['content_column']].tolist())
input_ids_list = [torch.tensor(  input_ids, dtype = torch.int) for input_ids in input_ids_list.input_ids]

logit_dict = analyze_logits(
    logit_tensor_sequence = [ soft( torch.from_numpy(token) ) for token in out] , #Out is a complete sequence
    input_token_ids = input_ids_list[0], ## SAMPLE ID
    tokenizer_fn = tokenizer
)

# %%
assert len(set(len(v) for v in logit_dict.values())) == 1, "All key array values in logit_dict do not have the same length"

# %% [markdown]
# #### Processing all the Batches

# %%
def process_logit_batches(tokenizer, tokenized_inputs, num_samples=10000, skip_first_token=True):
    """
    Process multiple saved logits files and extract:
    - top-1 max logit predictions,
    - top-1 min logit predictions,
    - actual logits for ground-truth tokens.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        tokenized_inputs (List[Tensor]): Tokenized input prompts (one per sample).
        num_samples (int): Number of samples to process.
        skip_first_token (bool): Whether to skip the first token prediction.

    Returns:
        Tuple of lists: (max_logit_predictions, min_logit_predictions, actual_logits)
    """
    max_logit_predictions = []
    min_logit_predictions = []
    actual_logit_scores = []

    softmax_fn = torch.nn.Softmax(dim=0)
    base_path = f"{params['callbacks_path']}/{params['current_model']}_q_{params['quantization']}"

    for sample_idx in range(num_samples):
        logits_file_path = f"{base_path}/logits_tensor[{sample_idx}]_batch[{sample_idx}].npy"
        logits_array = np.load(logits_file_path)[0]  # Shape: [sequence_length, vocab_size]

        # Apply softmax to each token’s logits
        normalized_logits = [softmax_fn(torch.from_numpy(token_logits)) for token_logits in logits_array]

        # Analyze logits using the unified function
        result = analyze_logits(
            logit_tensor_sequence=normalized_logits,
            input_token_ids=tokenized_inputs[sample_idx],
            tokenizer_fn=tokenizer,
            skip_first_token=skip_first_token
        )

        max_logit_predictions.append(result["max_cases"])
        min_logit_predictions.append(result["min_cases"])
        actual_logit_scores.append(result["actual_logits"])

        logging.info(f"Processed sample {sample_idx}")
        print(f"Processed sample {sample_idx}")

    return max_logit_predictions, min_logit_predictions, actual_logit_scores

# %%
input_ids_list = tokenizer.batch_encode_plus(df_dataset[params['dataset']['content_column']].tolist())
input_ids_list = [torch.tensor(  input_ids, dtype = torch.int) for input_ids in input_ids_list.input_ids]

# %%
max_logit_token_prompt, min_logit_token_prompt, actual_logit_token_prompt = process_logit_batches(
    tokenizer=tokenizer , tokenized_inputs=input_ids_list, 
    num_samples = len(df_dataset)
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
output_dir = f"{params['output_path']}/{params['current_model']}_q_{params['quantization']}"
create_folder(output_dir)
dataframe_to_save.to_json(f"{output_dir}/raw_logits.json", index=False)

# %% [markdown]
# #### Loss Retrieval

# %%
def batching_loss( size = dataframe_to_save.shape[0] ):
    output_dir = f"{params['callbacks_path']}/{params['current_model']}_q_{params['quantization']}"
    output_loss = []
    for current_batch in range(size):
        out = np.load(f"{output_dir}/_loss_batch[{current_batch}].npy")
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
dataframe_to_save.to_json(f"{output_dir}/raw_logits.json", index=False)

# %%
print("================================= PROCESS COMPLETED =================================")

# %%
del model
torch.cuda.empty_cache()
gc.collect()


