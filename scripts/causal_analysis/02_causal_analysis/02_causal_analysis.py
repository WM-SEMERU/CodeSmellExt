# %% [markdown]
# # Causal Analysis

# %%
from CodeSmells import pos_utils as pos_utils

# %%
def default_params(): 
    return {
        'input_path': '/workspaces/CodeSmells/data/extension/causality/dataset',
        'output_path': '/workspaces/CodeSmells/data/extension/causality/results',
        'causal_analysis': {
            'potential_outcomes': ['code_smell_psc_relative'], #code_smell_actual_prob_median, #code_smell_max_prob_median, #code_smell_min_prob_median, #code_smell_actual_prob_mean, #code_smell_max_prob_mean, #code_smell_min_prob_mean, #code_smell_psc_entropy, #code_smell_psc_relative  
            'common_causes': lambda causes: causes,
            'effect_modifiers' : lambda effect_modifiers: effect_modifiers,
            'instruments' : lambda instruments: instruments
        }, 
        ######## FOR CAUSAL ANALYSIS ########
        'features': {
            'syntactic' : ['complexity', 'n_ast_errors', 'ast_levels', 'n_whitespaces', 'n_words', 'vocab_size', 'nloc', 'token_counts', 'n_ast_nodes', 'n_identifiers'],
            'semantic' : [] + pos_utils.UNIVERSAL_SEMANTIC_TAGS,
        },
        'intervention' : {
            'type': 'T4', # 'T1' for generation_type, 'T2' for model_size, 'T3' for model_architecture, 'T4' for prompt
            'control': 'P1', # Each control is a baseline value from the selected intervention. (eg. 'curated' (base) for T1)
        },
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
    }
params = default_params()

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
from dowhy import CausalModel
import dowhy.datasets
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
import json
import os
import scipy.stats as stats

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ### Dataset loading

# %%
causal_hypothesis_df = pd.read_json(f"{params['input_path']}/{params['intervention']['type']}.json")

# %% [markdown]
# ### Causal Modeling

# %%
class RefutationResult:
    def __init__(self, refutation_result=None, new_effect=None):
        self.refutation_result = refutation_result
        self.new_effect = new_effect

# %%
def compute_ATE_and_refute(causal_model, identified_estimand, method_name, method_params={}):
    try:
        estimate = causal_model.estimate_effect(identified_estimand, method_name, method_params = method_params, test_significance=True)
    except Exception as e:
        print(f"======= ERROR COMPUTING CE for {method_name}: {e} ============")
        return {'method_name': method_name, 
            'estimated_effect': None,
            'refutation_placebo_permute' : None, 
            'refutation_unobserved_confounder' : None, 
            'refutation_subset' : None}
    
    refutation_placebo_permute = RefutationResult()
    refutation_unobserved_confounder = RefutationResult()
    refutation_subset = RefutationResult()
    refutation_random_common_cause = RefutationResult()
    
    try:
        refutation_placebo_permute = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute")
    except Exception as e:
        print(f"Error performing pacebo refutation - {e}")
    try: 
        refutation_unobserved_confounder = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="add_unobserved_common_cause")
    except Exception as e:
        print(f"Error performing unobserved covariate refutation - {e}")
    try:
        refutation_subset = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="data_subset_refuter")
    except Exception as e:
        print(f"Error performing subset refutation - {e}")
    try: 
        refutation_random_common_cause = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="random_common_cause")
    except Exception as e:
        print(f"Error performing random covariate refutation - {e}")
    

    return {'method_name': method_name, 
            'estimated_effect': estimate.value,
            'refutation_random_common_cause' : {'result' : refutation_random_common_cause.refutation_result, 'new_effect': refutation_random_common_cause.new_effect},
            'refutation_placebo_permute' : {'result' : refutation_placebo_permute.refutation_result, 'new_effect': refutation_placebo_permute.new_effect}, 
            'refutation_unobserved_confounder' : {'result' : refutation_unobserved_confounder.refutation_result, 'new_effect': refutation_unobserved_confounder.new_effect}, 
            'refutation_subset' : {'result' : refutation_subset.refutation_result, 'new_effect': refutation_subset.new_effect},}

def compute_correlations(input_corr_data, outout_corr_data):
    # Ensure both arrays have at least 2 elements
    if len(input_corr_data) < 2 or len(outout_corr_data) < 2:
        return {'pearson_corr': None, 'spearman_corr': None, 'kendall_corr': None}
    pearson_corr =  stats.pearsonr(input_corr_data, outout_corr_data)
    spearman_corr = stats.spearmanr(input_corr_data, outout_corr_data)
    kendall_corr = stats.kendalltau(input_corr_data, outout_corr_data)
    return {
        'pearson_corr': pearson_corr.statistic,
        'spearman_corr': spearman_corr.statistic,
        'kendall_corr': kendall_corr.statistic
    }
def compute_causal_effects(causal_model):
    causal_effects_df = pd.DataFrame(columns=['method_name', 'pearson_corr', 'spearman_corr', 'kendall_corr' ,'estimated_effect', 'refutation_placebo_permute', 'refutation_unobserved_confounder', 'refutation_subset'])
    ####### COMPUTE PEARSON
    #correlation_results = compute_correlations(list(causal_model._data[causal_model._common_causes].mean(axis=1)), list(causal_model._data[causal_model._outcome].mean(axis=1)))
    correlation_results = compute_correlations(causal_model._data['binary_treatment'].tolist(), list(causal_model._data[causal_model._outcome].mean(axis=1)))
    ####### COMPUTE ESTIMAND
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - propensity_score_matching
    causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_matching")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - propensity_score_stratification
    causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_stratification")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - backdoor.propensity_score_weighting
    causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_weighting")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - backdoor.propensity_score_weighting
    causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.generalized_linear_model", method_params={"glm_family": sm.families.Gaussian()})}
    
    return causal_effects_df.dropna()


# %%
def compute_causal_effect_for_binary_treatments(causal_hypothesis_df, outcomes):
    """
    Computes causal effects for each treatment (excluding control) in the given DataFrame.
    Returns a list of result DataFrames, one per treatment.
    """
    causal_effect_dfs = []
    for treatment in (t for t in causal_hypothesis_df['treatment'].unique() if t != params['intervention']['control']):
        print(f"=========== CAUSAL ANALYSIS FOR TREATMENT-{treatment}")
        # Extract treatment and control dataframes
        treatment_df = causal_hypothesis_df[causal_hypothesis_df['treatment'] == treatment].copy()
        control_df = causal_hypothesis_df[causal_hypothesis_df['treatment'] == params['intervention']['control']].copy()

        # Ensure treatment_df and control_df have the same size
        min_size = min(len(treatment_df), len(control_df))
        treatment_df = treatment_df.sample(n=min_size, random_state=42).reset_index(drop=True)
        control_df = control_df.sample(n=min_size, random_state=42).reset_index(drop=True)

        # Combine treatment and control
        intervention_df = pd.concat([treatment_df, control_df], ignore_index=True)

        # Set binary_treatment column
        intervention_df['binary_treatment'] = intervention_df['treatment'].map(lambda group: group == treatment)

        # Define causal model 
        causal_model = CausalModel(
            data=intervention_df,
            treatment=['binary_treatment'],
            outcome=outcomes,
            common_causes=params['causal_analysis']['common_causes'](params['features']['syntactic'] + params['features']['semantic']),
            effect_modifiers=params['causal_analysis']['effect_modifiers'](None),
            instruments=params['causal_analysis']['instruments'](None)
        )

        # Store results
        result_df = compute_causal_effects(causal_model)
        result_df['intervention'] = treatment
        causal_effect_dfs.append(result_df)
    return causal_effect_dfs

def compute_causal_effect_for_categorical_treatments(causal_hypothesis_df, outcomes):
    """
    Computes causal effects for categorical in the given DataFrame.
    """
    causal_effect_dfs = []

    # Compute the minimum group size
    group_sizes = causal_hypothesis_df['treatment'].value_counts()
    min_size = group_sizes.min()
    # Sample min_size rows from each treatment group
    balanced_causal_hypothesis_df = (causal_hypothesis_df
        .groupby('treatment', group_keys=False)
        .apply(lambda x: x.sample(n=min_size, random_state=42))
        .reset_index(drop=True))
    
    causal_model = CausalModel(
            data=balanced_causal_hypothesis_df,
            treatment=['treatment'],
            outcome=outcomes,
            common_causes=params['causal_analysis']['common_causes'](params['features']['syntactic'] + params['features']['semantic']),
            effect_modifiers=params['causal_analysis']['effect_modifiers'](None),
            instruments=params['causal_analysis']['instruments'](None))
    
     # Store results
    result_df = compute_causal_effects(causal_model)
    result_df['intervention'] = treatment
    causal_effect_dfs.append(result_df)
    return causal_effect_dfs
        
def compute_causal_effects_smells(causal_hypothesis_df, outcomes, binary_treatments=True):
    s_msg_id_effects = {}
    for s_msg_id in causal_hypothesis_df['s_msg_id'].unique():
        print(f"=========== CAUSAL ANALYSIS FOR CODE SMELL-{s_msg_id} ===========")
        # Subset for current s_msg_id
        causal_hypothesis_subset = causal_hypothesis_df[causal_hypothesis_df['s_msg_id'] == s_msg_id].copy()
        causal_effect_dfs = compute_causal_effect_for_binary_treatments(causal_hypothesis_subset, outcomes) if binary_treatments else compute_causal_effect_for_categorical_treatments(causal_hypothesis_subset, outcomes)
        s_msg_id_effects_df = pd.concat(causal_effect_dfs, ignore_index=True)
        s_msg_id_effects_df['s_msg_id'] = s_msg_id
        s_msg_id_effects[s_msg_id] = s_msg_id_effects_df
    return s_msg_id_effects

def compute_causal_effect_hypothesis(causal_hypothesis_df, outcomes, binary_treatments=True):
    causal_effect_dfs = compute_causal_effect_for_binary_treatments(causal_hypothesis_df, outcomes) if binary_treatments else compute_causal_effect_for_categorical_treatments(causal_hypothesis_df, outcomes)
    return pd.concat(causal_effect_dfs, ignore_index=True)

# %%
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %% [markdown]
# ### EXECUTE

# %%
def execute_analysis_per_smell(outcomes, causal_hypothesis_df):
    def run_and_store(outcome, binary_treatments, subfolder=""):
        analysis_type = "CATEGORICAL" if not binary_treatments else ""
        print(f"============================= COMPUTING ANALYSIS FOR OUTCOME-{outcome} {analysis_type} ==================================== ")
        causal_effect_outcome_dict = compute_causal_effects_smells(
            causal_hypothesis_df, [outcome], binary_treatments=binary_treatments
        )
        print(f"============================= STORING ANALYSIS FOR OUTCOME-{outcome} {analysis_type} ==================================== ")
        output_path = f"{params['output_path']}/{params['intervention']['type']}{subfolder}"
        create_folder(output_path)
        causal_effects_outcome_df = pd.concat(causal_effect_outcome_dict.values(), ignore_index=True)
        causal_effects_outcome_df.to_json(f"{output_path}/{outcome}.json")

    for outcome in outcomes:
        run_and_store(outcome, binary_treatments=False, subfolder="/categorial")
        run_and_store(outcome, binary_treatments=True)

# %%
def execute_analysis(outcomes, causal_hypothesis_df):
    def run_and_store(outcome, binary_treatments, subfolder="", suffix=""):
        analysis_type = " CATEGORICAL" if not binary_treatments else ""
        print(f"============================= COMPUTING ANALYSIS FOR OUTCOME-{outcome}{analysis_type} ==================================== ")
        causal_effects_outcome_df = compute_causal_effect_hypothesis(
            causal_hypothesis_df, [outcome], binary_treatments=binary_treatments
        )
        print(f"============================= STORING ANALYSIS FOR OUTCOME-{outcome}{analysis_type} ==================================== ")
        output_path = f"{params['output_path']}/{params['intervention']['type']}{subfolder}"
        create_folder(output_path)
        causal_effects_outcome_df.to_json(f"{output_path}/{outcome}{suffix}.json")

    for outcome in outcomes:
        run_and_store(outcome, binary_treatments=False, subfolder="/categorical", suffix="_ALL")
        run_and_store(outcome, binary_treatments=True, suffix="_ALL")

# %%
print("########################### Executing causal analysis per smell ###########################")
execute_analysis_per_smell(params['causal_analysis']['potential_outcomes'], causal_hypothesis_df)
print("########################### Executing causal analysis ALL smells ###########################")
execute_analysis(params['causal_analysis']['potential_outcomes'], causal_hypothesis_df)


