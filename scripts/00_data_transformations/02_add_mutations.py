# %% [markdown]
# ### Add mutatios for robustness testing

# %%
def default_params(): 
    return {
        'bert_model': 'microsoft/codebert-base-mlm',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'dataset_path' : '/workspaces/CodeSmells/semeru-datasets/code_smells',
        'sampling_size' : 500
    }
params = default_params()


# %% [markdown]
# ### Imports

# %%
import torch
import gc
import json
import subprocess
import os
import glob

# %%
import pandas as pd
import numpy as np
from CodeSmells import semantic_preserving_transformations as trans

# %%
from transformers import logging

logging.set_verbosity_error()


# %% [markdown]
# ### Read Dataframe

# %%
dataset_df = pd.read_json(f"{params['dataset_path']}/curated_{params['sampling_size']}.json")

# %%
dataset_df = dataset_df

# %% [markdown]
# ### Add mutations

# %%
def run_pylint_analysis(code):
    temp_file_path = f"{params['cache_dir']}/pylint/temp.py"
     # Save the code to a temporary file with UTF-8 encoding
    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
        temp_file.write(code)
    command = [
        'pylint',
        temp_file_path,
        '--output-format=json'
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    full_analysis = json.loads(result.stdout) if result.stdout else []

    # Read the code again to extract lines, using UTF-8 encoding
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    os.remove(temp_file_path)  # Clean up the temporary file

    simplified_analysis = []
    for issue in full_analysis:
        line = int(issue['line']) if issue['line'] is not None else 0
        end_line = int(issue.get('endLine', line)) if issue.get('endLine', line) is not None else line
        start_line = line - 1
        end_line = end_line - 1

        issue_code = ''.join(lines[start_line:end_line + 1]).strip() if start_line >= 0 and end_line >= 0 and start_line <= end_line else ""

        simplified_analysis.append({
            'code' : code,
            'msg_id': issue['message-id'],
            'line': issue['line'],
            'column': issue['column'],
            'end_line': issue.get('endLine', issue['line']),
            'end_column': issue['endColumn'],
            'code_smell': issue_code
        })

    return simplified_analysis

# %%
def compute_end_column(smell):
    if pd.notna(smell['end_column']):
        return smell['end_column']
    code_lines = smell['code'].splitlines()
    
    # Validate that end_line is within the bounds of code_lines
    if smell['end_line'] < 0 or smell['end_line'] >= len(code_lines):
       raise IndexError("end_line is out of range.")
    
    return len(code_lines[smell['end_line']])


# %%
def extract_substring(code_string, start_line, start_column, end_line, end_column):
    # Split the string into individual lines.
    lines = code_string.splitlines()
    # Validate the provided indices.
    if start_line < 0 or start_line >= len(lines):
        raise IndexError("start_line is out of range.")
    if end_line < 0 or end_line >= len(lines):
        raise IndexError("end_line is out of range.")
    # Case when the substring is within a single line.
    if start_line == end_line:
        return lines[start_line][start_column:end_column]
    # Extract parts from multiple lines.
    # 1. Extract from the start line starting at start_column.
    extracted_lines = [lines[start_line][start_column:]]
    # 2. Add all the lines between the start and end lines (if any).
    for line in lines[start_line + 1 : end_line]:
        extracted_lines.append(line)
    # 3. Extract from the end line up to end_column.
    extracted_lines.append(lines[end_line][:end_column])
    # Join the parts with newline characters.
    return "\n".join(extracted_lines)

# %%
def find_first_matching_object(row, transformation_column:str):
    return next((obj for obj in row[transformation_column] if obj.get('msg_id') == row['s_msg_id'] and obj.get('line') >= row['s_line']), None)


# %%
def fix_smell_pos_values(smell):
    smell['line'] -= 1
    if pd.isna(smell['end_line']):
        smell['end_line'] = smell['line']
    else: 
        smell['end_line'] -= 1
    smell['end_column'] = compute_end_column(smell)
    smell['code_smell'] = extract_substring(smell['code'], smell['line'], smell['column'], smell['end_line'], smell['end_column'])
    return smell
        

# %%
def add_transformation(df, transformation_name, transformation_function, filter_unique):
    df = df.copy()
    df[transformation_name] = df.apply(lambda row: [fix_smell_pos_values(smell) for smell in run_pylint_analysis(transformation_function(row['code']))], axis=1)
    df[transformation_name] = df.apply(lambda row: find_first_matching_object(row, transformation_name), axis=1)
    ###filter by change in code
    if filter_unique: df[transformation_name] = df.apply(lambda row: row[transformation_name] if (row[transformation_name] and row[transformation_name]['code_smell']!= row['s_code']) else None, axis=1)
    df = df.reset_index(drop=True)
    return df


# %%
def execute_rename_variable_2(code:str):
    return trans.rename_variable_2(code, params['bert_model'], params['cache_dir'])

# %%
dataset_df = add_transformation(dataset_df, 'RenameVariable-1', trans.rename_variable_1, False)
dataset_df = add_transformation(dataset_df, 'RenameVariable-2', execute_rename_variable_2, False)
dataset_df = add_transformation(dataset_df, 'Add2Equal', trans.add_2_equal, False)
dataset_df = add_transformation(dataset_df, 'SwitchEqualExp', trans.switch_equal_exp, False)
dataset_df = add_transformation(dataset_df, 'InfixDividing', trans.infix_dividing, False)
dataset_df = add_transformation(dataset_df, 'SwitchRelation', trans.switch_relation, False)

# %%
dataset_df

# %%
#row_number = 18
#print(dataset_df.loc[row_number]['s_code'])
#print(dataset_df.loc[row_number]['RenameVariable-2']['code_smell'])


# %% [markdown]
# ### Store the data

# %%
dataset_df.to_json(f"{params['dataset_path']}/transformed_curated_{params['sampling_size']}.json", index=False)


