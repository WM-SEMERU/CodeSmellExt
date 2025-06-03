# %% [markdown]
# ## Dataset Preprocessing for Causal Analysis

# %%
from CodeSmells import pos_utils as pos_utils

# %%
def default_params(): 
    return {
        'output_path' : '/workspaces/CodeSmells/data/extension/causality/dataset',
        ######## CAUSAL ANALYSIS ########
        'features': {
            'syntactic' : ['complexity', 'n_ast_errors', 'ast_levels', 'n_whitespaces', 'n_words', 'vocab_size', 'nloc', 'token_counts', 'n_ast_nodes', 'n_identifiers'],
            'semantic' : [] + pos_utils.UNIVERSAL_SEMANTIC_TAGS,
        },
        'treatments': {
            #T1
            'generation_type' : ['greedy_search', 'beam_search', 'sampling', 'contrastive_search', 'top_k_sampling', 'top_p_sampling'], ### same model
            #T2
            'model_size' : ['S1', 'S2', 'S3', 'S4'], #same architecture
            #T3
            'model_architecture' : ['M1', 'M2', 'M3', 'M4'], #same size
            #T4
            'prompt' : ['P1', 'P2', 'P3', 'P4'],
        },
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
    }
params = default_params()

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import os

# %% [markdown]
# ### Append Semantic Features

# %%
def append_semantic_features(df, semantic_features: list):
    """
    Appends semantic features to the DataFrame
    """
    def update_row(row):
        row_semantic_features = pos_utils.count_pos_tags(row['code'])
        for feature in semantic_features:
            if feature in row_semantic_features:
                row[feature] = row_semantic_features[feature]
        return row


    updated_df = df.copy()
    for feature in semantic_features:
        updated_df[feature] = 0
    updated_df = updated_df.apply(update_row, axis=1)
    return updated_df

# %%
def keep_shared_columns(dfs):
    """
    Given a list of DataFrames, returns a new list where each DataFrame only contains columns shared by all.
    """
    if not dfs:
        return []
    shared_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        shared_cols &= set(df.columns)
    shared_cols = list(shared_cols)  # Convert set to list for pandas indexing
    return [df.loc[:, shared_cols].copy() for df in dfs]

# %% [markdown]
# ### Dataset Loading

# %%
def load_json_dfs_from_dir(dir_path):
    """
    Recursively loads all .json files in the given directory into a dictionary of pandas DataFrames.
    Each key is the name of the closest parent folder containing the .json file.
    If multiple files exist in the same folder, later ones will overwrite earlier ones.
    """
    dfs = {}
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                parent_folder = os.path.basename(os.path.dirname(file_path))
                try:
                    df = pd.read_json(file_path)
                    dfs[parent_folder] = df
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
    return dfs

# %% [markdown]
# ### Execute

# %%
### READ ALIGNMENTS
print("Loading alignments...")
sampling_dict = load_json_dfs_from_dir('/workspaces/CodeSmells/data/extension/generation/alignments')
pipeline_dict = load_json_dfs_from_dir('/workspaces/CodeSmells/data/extension/pipeline/alignments')
prompts_dict = load_json_dfs_from_dir('/workspaces/CodeSmells/data/extension/prompts/alignments')

# %%
############ DEFINE LISTS OF TREATMENTS
# generation_type
generation_type_dfs = keep_shared_columns([df.assign(treatment=key) for key, df in sampling_dict.items()])
# model_size
model_size_dfs = keep_shared_columns([df.assign(treatment=key.split('_')[0]) for key, df in pipeline_dict.items() if 'S' in key])
# model_architecture
model_architecture_dfs = keep_shared_columns([df.assign(treatment=key.split('_')[0]) for key, df in pipeline_dict.items() if 'M' in key])
# prompt
prompt_dfs = keep_shared_columns([df.assign(treatment=key) for key, df in prompts_dict.items()])

# %%
# Combine all DataFrames from all lists
all_dfs = generation_type_dfs + model_size_dfs + model_architecture_dfs + prompt_dfs

# Find shared columns across all DataFrames
if all_dfs:
    shared_cols = set(all_dfs[0].columns)
    for df in all_dfs[1:]:
        shared_cols &= set(df.columns)
    shared_cols = list(shared_cols)  # Convert to list for indexing

    # Update each list so all DataFrames have only the shared columns
    generation_type_dfs = [df.loc[:, shared_cols].copy() for df in generation_type_dfs]
    model_size_dfs = [df.loc[:, shared_cols].copy() for df in model_size_dfs]
    model_architecture_dfs = [df.loc[:, shared_cols].copy() for df in model_architecture_dfs]
    prompt_dfs = [df.loc[:, shared_cols].copy() for df in prompt_dfs]

# %%
# Check if all DataFrames in generation_type_dfs have the same columns
columns_list = [set(df.columns) for df in generation_type_dfs]
all_same = all(cols == columns_list[0] for cols in columns_list)
print("All DataFrames have the same columns:", all_same)

# Optionally, print columns that are different
if not all_same:
    for idx, cols in enumerate(columns_list):
        print(f"DataFrame {idx} columns ({len(cols)}): {sorted(list(cols))}")

# %%
# Check if all DataFrames from all lists contain the same columns

all_dfs = generation_type_dfs + model_size_dfs + model_architecture_dfs + prompt_dfs

if all_dfs:
    columns_list = [set(df.columns) for df in all_dfs]
    all_same = all(cols == columns_list[0] for cols in columns_list)
    print("All DataFrames from all lists have the same columns:", all_same)
    if not all_same:
        for idx, cols in enumerate(columns_list):
            print(f"DataFrame {idx} columns ({len(cols)}): {sorted(list(cols))}")
else:
    print("No DataFrames to compare.")

# %%
#### APPEND SEMANTIC FEATURES
print("Appending semantic features to DataFrames...")
generation_type_dfs = [append_semantic_features(df, params['features']['semantic']) for df in generation_type_dfs]
model_size_dfs = [append_semantic_features(df, params['features']['semantic']) for df in model_size_dfs]
model_architecture_dfs = [append_semantic_features(df, params['features']['semantic']) for df in model_architecture_dfs]
prompt_dfs = [append_semantic_features(df, params['features']['semantic']) for df in prompt_dfs]

# %% [markdown]
# ### Store Dataframes

# %%
def save_dfs_as_json_array(dfs, output_path):
    """
    Saves a list of DataFrames as a single JSON array to the specified output path.
    """
    if not dfs:
        print("No DataFrames to save.")
        return
    combined_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_json(output_path, orient='records', lines=False)
    print(f"Saved {len(dfs)} DataFrames as JSON array to {output_path}")

# %%
print(f"Saving {len(generation_type_dfs)} DataFrames for generation_type")
save_dfs_as_json_array(generation_type_dfs, params['output_path'] + '/T1.json')
print(f"Saving {len(model_size_dfs)} DataFrames for model_size")
save_dfs_as_json_array(model_size_dfs, params['output_path'] + '/T2.json')
print(f"Saving {len(model_architecture_dfs)} DataFrames for model_architecture")
save_dfs_as_json_array(model_architecture_dfs, params['output_path'] + '/T3.json')
print(f"Saving {len(prompt_dfs)} DataFrames for prompt")
save_dfs_as_json_array(prompt_dfs, params['output_path'] + '/T4.json')


