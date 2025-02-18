# %% [markdown]
# ### Add mutatios for robustness testing

# %%
# Transformation names: 'RenameVariable-1' 'RenameVariable-2'
#                       'Add2Equal' 'SwitchEqualExp''InfixDividing' 
#                       'SwitchRelation' 
def default_params(): 
    return {
        'bert_model': 'microsoft/codebert-base-mlm',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'dataset_path' : '/workspaces/CodeSmells/semeru-datasets/code_smells',
        'sampling_size' : 500, 
        'transformation_name' : 'SwitchRelation'
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
import re
import lizard

# %%
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

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
#dataset_df = dataset_df[:20]

# %% [markdown]
# ### Add mutations

# %%
def run_pylint_analysis(code):
    temp_file_path = f"{params['cache_dir']}/pylint/temp_{params['transformation_name']}.py"
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

# %% [markdown]
# ### Disentangle Transformation

# %%
class TreeSitterManager():
    def __init__(self, lang):
        #self.language = self.get_language(lang)
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        #self.parser.set_language(self.language)

    def get_ast_errors_and_deep(self, code):
        node_tree = self.parser.parse(bytes(code, "utf8"))
        return self.__detect_ast_errors_and_deep(node_tree.root_node, set())

    def __detect_ast_errors_and_deep(self, node_root, identifier_set = set(), level=0, max_level=0, count = 0):
        """Traverses the tree catch errors and evaluate tree levels"""
        # if not node_root.has_error:
        #     return [], 0
        counter = node_root.child_count

        results = []
        if node_root.type == "ERROR":
            results.append(node_root.text.decode("utf-8"))
        elif node_root.type == "identifier":
            identifier_set.add(node_root.text)
        level += 1
        for n in node_root.children:
            x, identifier_set, y, max_level, count = self.__detect_ast_errors_and_deep(n, identifier_set, level, max_level)
            max_level = max(y, max_level)
            counter += count
            if len(x) > 0:
                results.extend(x)

        return results, identifier_set, max_level, level, counter

# %%
def analyze_method(code_string):
    temp_file_path = f"{params['cache_dir']}/lizard/temp_{params['transformation_name']}.py"
    # Analyze the code using lizard
    analysis = lizard.analyze_file.analyze_source_code(temp_file_path, code_string)
    
    # Extract function details
    functions = []
    for function in analysis.function_list:
        functions.append({
            "fun_name": function.name,
            "complexity": function.cyclomatic_complexity,
            "nloc": function.nloc,
            "token_counts": function.token_count
        })
    
    return functions[0] if functions else None

# %%
def disentangle_transformation(entangled_df):
    # Initialize the AST error detector (assuming language is always Python)
    ast_error_detector = TreeSitterManager("python")
    t_name = params['transformation_name']  # transformation column name

    # Copy shared columns
    shared_columns = ['id', 'commit_id', 'repo', 'path', 'file_name', 
                      'commit_message', 'url', 'language', 'category']
    transformation_df = entangled_df[shared_columns].copy()

    # Extract transformation details from the JSON field
    transformation_df['code'] = entangled_df[t_name].apply(lambda t: t.get('code') if t else None)
    transformation_df['s_msg_id'] = entangled_df[t_name].apply(lambda t: t.get('msg_id') if t else None)
    transformation_df['s_line'] = entangled_df[t_name].apply(lambda t: t.get('line') if t else None)
    transformation_df['s_column'] = entangled_df[t_name].apply(lambda t: t.get('column') if t else None)
    transformation_df['s_end_line'] = entangled_df[t_name].apply(lambda t: t.get('end_line') if t else None)
    transformation_df['s_end_column'] = entangled_df[t_name].apply(lambda t: t.get('end_column') if t else None)
    transformation_df['s_code'] = entangled_df[t_name].apply(lambda t: t.get('code_smell') if t else None)

    # Compute simple code metrics
    transformation_df['n_whitespaces'] = transformation_df['code'].apply(lambda code: code.count(' ') if code else None)
    transformation_df['n_words'] = transformation_df['code'].apply(lambda code: len(code.split()) if code else None)
    transformation_df['vocab_size'] = transformation_df['code'].apply(lambda code: len(set(code.split())) if code else None)

    # Compute lizard and AST metrics in one pass per row
    def compute_metrics(row):
        code = row['code']
        if not code:
            return pd.Series({
                'fun_name': None,
                'complexity': None,
                'nloc': None,
                'token_counts': None,
                'ast_errors': None,
                'ast_levels': None,
                'n_ast_nodes': None,
                'n_ast_errors': None,
                'n_identifiers': None
            })

        # Lizard analysis
        lizard_result = analyze_method(code)
        fun_name = lizard_result.get('fun_name') if lizard_result is not None else None
        complexity = lizard_result.get('complexity')  if lizard_result is not None else None
        nloc = lizard_result.get('nloc') if lizard_result is not None else None
        token_counts = lizard_result.get('token_counts') if lizard_result is not None else None

        # AST analysis
        ast_errors, identifier_set, ast_deep, level, count = ast_error_detector.get_ast_errors_and_deep(code)
        ast_levels = ast_deep
        n_ast_nodes = count
        n_ast_errors = len(ast_errors) if ast_errors is not None else None
        n_identifiers = len(identifier_set) if identifier_set is not None else None

        return pd.Series({
            'fun_name': fun_name,
            'complexity': complexity,
            'nloc': nloc,
            'token_counts': token_counts,
            'ast_errors': ast_errors,
            'ast_levels': ast_levels,
            'n_ast_nodes': n_ast_nodes,
            'n_ast_errors': n_ast_errors,
            'n_identifiers': n_identifiers
        })

    metrics_df = transformation_df.apply(compute_metrics, axis=1)
    transformation_df = pd.concat([transformation_df, metrics_df], axis=1)
    return transformation_df[pd.notnull(transformation_df['code'])].reset_index(drop=True)

# %% [markdown]
# ### Transformations Mapping

# %%
def default_transformations(): 
    return {
        'RenameVariable-1': trans.rename_variable_1, 
        'RenameVariable-2' : execute_rename_variable_2,
        'Add2Equal' : trans.add_2_equal, 
        'SwitchEqualExp' : trans.switch_equal_exp, 
        'InfixDividing' : trans.infix_dividing,
        'SwitchRelation' : trans.switch_relation
    }
transformation_map = default_transformations()


# %% [markdown]
# ### Execute

# %%
print(f"=========================== Transformation {params['transformation_name']} started =============================")
dataset_df = add_transformation(dataset_df, params['transformation_name'], transformation_map[params['transformation_name']], False)
print(f"=========================== Transformation {params['transformation_name']} finished =============================")

# %%
print(f"=========================== Saving entangled {params['transformation_name']} transformation =============================")
dataset_df.to_json(f"{params['dataset_path']}/entangled_{params['transformation_name']}_{params['sampling_size']}.json", index=False)

# %%
print(f"=========================== Disentaglement {params['transformation_name']} started =============================")
result_df = disentangle_transformation(dataset_df)
print(f"=========================== Disentaglement {params['transformation_name']} finished =============================")

# %%
result_df.head(5)

# %% [markdown]
# ### Store the data

# %%
print(f"=========================== Saving disentangled {params['transformation_name']} transformation =============================")
result_df.to_json(f"{params['dataset_path']}/transformed_{params['transformation_name']}_{params['sampling_size']}.json", index=False)


