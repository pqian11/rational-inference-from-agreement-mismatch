import pandas as pd 
import json
import numpy as np
from helper import *


def export_trial_df(data, trial_conditions, e_types, stimuli, group2idx, include_items=None, only_mismatch_resolving_trials=False, verbose=False):
    df = []

    trial_condition_set = set(trial_conditions)

    target_e_type_set = set(e_types)

    for i in range(len(data)):
        subject_data = data[i]
        subject_info = json.loads(subject_data[1]['responses'])
        if verbose:
            print('subject_{}'.format(i), subject_info['Q0'])

        for trial in subject_data[1:-1]:
            if trial['trial_type'] != 'text-editing':
                continue

            trial_condition = trial['condition']

            if trial_condition in trial_condition_set: # Only process target trials
                answer = trial['answer']
                group = trial['group']
                is_skipped = trial['is_skipped'] if 'is_skipped' in trial else False 
                stimulus_idx = group2idx[group]

                # Exclude a problematic item
                if stimulus_idx == 55:
                    # print(group, stimulus_idx)
                    continue
            
                if (include_items is not None) and stimulus_idx not in include_items:
                    continue
   
                if is_skipped:
                    inferred_e = 'skip'
                else:
                    # Map sentence to feature-based representation
                    inferred_hyp = map_to_feature(answer, stimuli[group2idx[group]])

                    if inferred_hyp == trial_condition:
                        inferred_e = 'no_change'   
                    else:
                        # Identify error type based on input and answer
                        inferred_e = identify_error_type(trial_condition, inferred_hyp)
                    
                if only_mismatch_resolving_trials:
                    if inferred_e in target_e_type_set:    
                        df.append([inferred_e, stimulus_idx, trial_condition, i, answer])
                else:
                    df.append([inferred_e, stimulus_idx, trial_condition, i, answer])

    df = pd.DataFrame(np.array(df), columns =['e_type', 'item', 'trial_condition', 'subject', 'response'])
    return df


# Load stimuli
infpath = 'data/prior_norming_stimuli.txt'
df = pd.read_csv(infpath)

stimuli = []
group2idx = {}

for i, row in df.iterrows():
    if i >= 57:
        # Exclude filler items in the norming study
        continue
        
    N1s = row['N1'].split('/')
    N2s = row['N2'].split('/')
    preds = row['predicate_sg_pl'].split('/')

    context = row['context'].format(preposition=row['preposition'], predicate='%%')
    norming_context = row['context'].format(preposition=row['preposition'], predicate=row['predicate'])

    group = N1s[0] + '_' + N2s[0]
    choices = [N1s, N2s, preds]

    stimuli.append({'group':group, 'choices':choices, 'stimulus':context, 'norming_stimulus': norming_context})
    group2idx[group] = i


EXP_NAMES = ['exp1', 'exp2']
study_index = ['I', 'II']

for exp_idx, EXP_NAME in enumerate(EXP_NAMES):
    data = json.load(open('data/{}_participant_data.json'.format(EXP_NAME)))

    trial_conditions = ['SSP', 'SPP', 'PSS', 'PPS']
    error_types = ['subj', 'verb', 'subj+local', 'verb+local']

    e2i = dict([(e, i) for i, e in enumerate(error_types)])
    cond2i = dict([(cond, i) for i, cond in enumerate(trial_conditions)])

    # Export dataframe of all the target trials
    df = export_trial_df(data, trial_conditions, error_types, stimuli, group2idx, include_items=None)
    print(df)
    df.to_csv('data/{}_df_target_trials.csv'.format(EXP_NAME))

    # Export dataframe of the target trials in which participants resolves the agreement mismatch
    df = export_trial_df(data, trial_conditions, error_types, stimuli, group2idx, 
        include_items=None, only_mismatch_resolving_trials=True)
    print(df)
    df.to_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(EXP_NAME))
