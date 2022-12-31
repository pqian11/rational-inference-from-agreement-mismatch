import pandas as pd
import numpy as np
import json


def pprint_error_count_by_condition(error_freq_count_by_condition):
    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Condition', *e_types))
    for cond in conditions:
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format(cond, *[error_freq_count_by_condition[cond][e_type] for e_type in e_types] ))  


def extract_cond_type(condition):
    tokens = condition.split('_')
    return tokens[1][0] + tokens[2][0] + '_'


def get_error_type_count_from_Ryskin_et_al_data(fpath):  
    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']

    error_freq_count_by_condition = {}
    for cond in conditions:
        error_freq_count_by_condition[cond] = {}
        for e_type in e_types:
            error_freq_count_by_condition[cond][e_type] = 0  

    df = pd.read_csv(fpath)
    for row_idx, row in df.iterrows():
        cond = extract_cond_type(row['Condition'])
        if row['First_Noun_Error'] == 1 and row['Verb_Agreement_Error'] == 1 and row['Second_Noun_Error'] != 1: 
            error_freq_count_by_condition[cond]['subject'] += 1
        elif row['First_Noun_Error'] != 1 and row['Verb_Agreement_Error'] == 1 and row['Second_Noun_Error'] != 1: 
            error_freq_count_by_condition[cond]['verb'] += 1
        elif row['First_Noun_Error'] == 1 and row['Verb_Agreement_Error'] == 1 and row['Second_Noun_Error'] == 1:
            error_freq_count_by_condition[cond]['subject+local'] += 1
        elif row['First_Noun_Error'] != 1 and row['Verb_Agreement_Error'] == 1 and row['Second_Noun_Error'] == 1:
            error_freq_count_by_condition[cond]['verb+local'] += 1
        else:
            pass
    return error_freq_count_by_condition


def flip_feature(x):
    if x == 'S':
        return 'P'
    elif x == 'P':
        return 'S'
    else:
        raise NotImplementedEroor


def get_intent_before_self_revision(response, is_subj_revision, is_loc_revision, is_verb_revision):
    rs = [c for c in response]
    if is_subj_revision == 1:
        rs[0] = flip_feature(response[0])
    if is_loc_revision == 1:
        rs[1] = flip_feature(response[1])    
    if is_verb_revision == 1:
        rs[2] = flip_feature(response[2])    
    return ''.join(rs)


conditions = ['SS_', 'SP_', 'PS_', 'PP_']
e_types = ['subject', 'verb', 'subject+local', 'verb+local']


# Analyze experiment 1a from Ryskin et al.
dir_path = 'data/Ryskin_et_al'
fpath = '{dir_path}/Expt{exp_index}/expt{exp_index}_results.csv'.format(dir_path=dir_path, exp_index=1)
e_dict = get_error_type_count_from_Ryskin_et_al_data(fpath)
print('Ryskin et al:')
pprint_error_count_by_condition(e_dict)
print()
json.dump(e_dict, open('data/ryskin_et_al_exp1a_error_count.json', 'w'), indent=4)


# Analyze experiments from Kandel et al. (2022)
e_map = {
    'SS_':{'PSS':'subject', 'SSP':'verb', 'PPS':'subject+local', 'SPP':'verb+local'},
    'SP_':{'PPS':'subject', 'SPP':'verb', 'PSS':'subject+local', 'SSP':'verb+local'},
    'PS_':{'SSP':'subject', 'PSS':'verb', 'SPP':'subject+local', 'PPS':'verb+local'},
    'PP_':{'SPP':'subject', 'PPS':'verb', 'SSP':'subject+local', 'PSS':'verb+local'},
}

exp_names = ['exp1', 'exp2']
for exp_name in exp_names:
    path = 'data/Kandel_et_al/{}_data.csv'.format(exp_name)
    df = pd.read_csv(path)

    e_dict = {}
    for cond in conditions:
        e_dict[cond] = {}
        for e_type in e_types:
            e_dict[cond][e_type] = 0
            
    for row_idx, row in df.iterrows():
        cond = row['item'][:2]+'_'
        response = row['response_structure']        
        response = get_intent_before_self_revision(response, row['subj_revision'], row['loc_revision'], row['verb_revision'])    
        
        if response[-1] == response[0]:
            continue
        else: 
            e_type = e_map[cond][response]
            e_dict[cond][e_type] += 1

    print('Kandel et al. (2022) {}:'.format(exp_name.title()))
    pprint_error_count_by_condition(e_dict)
    print()
    json.dump(e_dict, open('data/kandel_et_al_{}_error_count.json'.format(exp_name), 'w'), indent=4)

