import pandas as pd 
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
from helper import *


def display_subject_data(subject_data, trial_conditions, error_types):
    print('-'*50)
    print('{0: >5} {1: >10} {2: >10} {3: >10} {4: >10}'.format('', *error_types))
    for k, condition in enumerate(trial_conditions):
        print('{0: >5} {1: >10.0f} {2: >10.0f} {3: >10.0f} {4: >10.0f}'.format(condition, *list(subject_data[k])))
    print('-'*50)
    print('{: >5} {: >10.0f} {: >10.0f} {: >10.0f} {: >10.0f}'.format('Total', *list(np.sum(subject_data, axis=0))))
    print('-'*50)
    return


def extract_e_type_count(data, trial_conditions, e_types, stimuli, group2idx, include_items=None, verbose=False):
    trial_condition_set = set(trial_conditions)

    e2i = dict([(e, i) for i, e in enumerate(e_types)])
    cond2i = dict([(cond, i) for i, cond in enumerate(trial_conditions)])

    e_type_count_by_item_condition = np.zeros((len(stimuli), len(trial_conditions), len(e_types)))
    e_type_count_by_subject_condition = np.zeros((len(data), len(trial_conditions), len(e_types)))

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
                    
                if inferred_e == 'no_change' or inferred_e == 'skip' or inferred_e == 'other':
                    if verbose:
                        print('{} --> {}'.format(trial_condition, inferred_hyp), '{0: >11}'.format(inferred_e), '[{}]'.format(group), answer)
                    # continue
                    if stimulus_idx == 55:
                        print('{} --> {}'.format(trial_condition, inferred_hyp), '{0: >11}'.format(inferred_e), '[{}]'.format(group), answer)


                if inferred_e in e2i:
	                e_type_count_by_item_condition[stimulus_idx, cond2i[trial_condition], e2i[inferred_e]] += 1
	                e_type_count_by_subject_condition[i, cond2i[trial_condition], e2i[inferred_e]] += 1

#             print('{} --> {}'.format(trial_condition, inferred_hyp), '{0: >11}'.format(inferred_e), '[{}]'.format(group), answer)

        if verbose:
            print()
            display_subject_data(e_type_count_by_subject_condition[i], trial_conditions, e_types)  
            print()
        
    return e_type_count_by_subject_condition, e_type_count_by_item_condition


def plot_priors(pi_list, savepath=None):
	pi_mean = np.mean(pi_list, axis=0)
	pi_err = np.std(pi_list, axis=0)*1.96

	plt.figure(figsize=(5,3))

	for i, pi in enumerate(pi_list):
		# Exclude a problematic item
		if i == 55:
			continue
		plt.plot(pi, 'ko-', mfc='none', alpha=0.1)
	    
	plt.errorbar(range(4), pi_mean, yerr=pi_err, fmt='o-')
	plt.ylim(0,0.45)
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)  
	ax.set_xticks(range(4))
	ax.set_xticklabels(['SS_', 'SP_', 'PS_', 'PP_'])
	plt.ylabel('Estimated prior prob.')
	if savepath:
		plt.savefig(savepath, bbox_inches='tight')
	plt.show()


def plot_inferred_error_frequency(normalized_error_type_fdist_by_condition_all, xlabels, title=None, 
                                  figsize=(10, 3), seed=1001, legend_loc='center left', bbox_to_anchor=(1.02, 0.5), legend_ncol=1, savepath=None):
    # Set seed of the random jitter for reprocudible figures
    np.random.seed(seed)
#     colors = ['gold', 'magenta', 'royalblue', 'limegreen']
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    n_condition = len(normalized_error_type_fdist_by_condition_all)
    n_error = len(normalized_error_type_fdist_by_condition_all[0][0])
    colors = [plt.cm.tab10(i) for i in range(n_condition)]
    fontsize = 12
#     fontsize = 16
    
    for i in range(n_condition):
#         print(np.mean(normalized_error_type_fdist_by_condition_all[i], axis=0))
#         print(np.std(normalized_error_type_fdist_by_condition_all[i], axis=0))

        xs = [e_idx+0.2*i for e_idx in range(n_error)]
        ys = np.mean(normalized_error_type_fdist_by_condition_all[i], axis=0)
        n_obs = len(normalized_error_type_fdist_by_condition_all[i])
        yerrs = 1.96*np.std(normalized_error_type_fdist_by_condition_all[i], axis=0)/np.sqrt(n_obs)
        plt.bar(xs, ys, width=0.12, label=trial_conditions[i], yerr=yerrs, color=colors[i])
        
        # Plot each data point with small jitter along x-axis
        for j in range(len(normalized_error_type_fdist_by_condition_all[i])):
            xjitter = np.random.uniform()*0.1-0.05
            xs = [e_idx+0.2*i+xjitter for e_idx in range(n_error)]
            ys = normalized_error_type_fdist_by_condition_all[i][j]
            plt.plot(xs, ys, 'ko', alpha=0.2, mfc='none', markersize=4.5, markeredgewidth=0.7)
                    
    plt.legend(ncol=legend_ncol, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    ax.set_xticks([x + 0.3 for x in range(n_error)])
    ax.set_xticklabels(xlabels, fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    plt.ylabel('Frequency', fontsize=fontsize*0.85)
    ax.tick_params(axis='y', which='major', labelsize=fontsize*0.85)
    
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_error_correction_preference_panel(error_type_count_by_subject_condition, other_edit_type_count_by_subject_condition, 
    trial_conditions, error_types, study_index, figsize=(10, 2.5), savepath=None):
    error_type_freq_by_subject_condition = np.zeros(error_type_count_by_subject_condition.shape)
    for k, subject_e_type_count in enumerate(error_type_count_by_subject_condition):
        e_type_freq = np.zeros((len(trial_conditions), len(error_types)))
        for i in range(len(trial_conditions)):
            if np.sum(subject_e_type_count[i]) >= 1:
                e_type_freq[i] = normalize(subject_e_type_count[i])
        error_type_freq_by_subject_condition[k] = e_type_freq

    normalized_error_type_fdist_by_condition_all = [[], [], [], []]

    for normalized_fdist in error_type_freq_by_subject_condition:
        for i in range(len(trial_conditions)):
            if np.sum(normalized_fdist[i]) > 0:
                normalized_error_type_fdist_by_condition_all[i].append(normalized_fdist[i])

    xlabel_names = ['subject', 'verb', 'subject+local', 'verb+local']

    normalized_other_edit_freq_all = []
    for subject_idx, subject_other_edit_count in enumerate(other_edit_type_count_by_subject_condition):
        subject_other_edit_total_count = np.sum(subject_other_edit_count, axis=1)
        subject_target_edit_total_count = np.sum(error_type_count_by_subject_condition[subject_idx], axis=1)
        subject_total_count = subject_other_edit_total_count + subject_target_edit_total_count
        normalized_other_edit_freq_all.append(np.divide(subject_other_edit_total_count, subject_total_count))

        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 2.8], wspace=0.2, hspace=0)

    # Plot the frequency of skip/no change/other edits given mismatch condition
    ax = fig.add_subplot(gs[0])
    np.random.seed(100)  # Set seed of the random jitter for reprocudible figures
    yerrs = 1.96*np.std(normalized_other_edit_freq_all, axis=0)/np.sqrt(len(normalized_other_edit_freq_all))
    ax.bar(np.arange(4), np.mean(normalized_other_edit_freq_all, axis=0), yerr=yerrs, 
            width=0.5, color='none', edgecolor='k', error_kw=dict(ecolor='r', capsize=4))
    for subject_other_edit_freq_given_condition in normalized_other_edit_freq_all:
        xjitter = np.random.uniform()*0.2-0.1
        ax.plot(np.arange(4)+xjitter, subject_other_edit_freq_given_condition, 'ko', mfc='none', 
            alpha=0.3, markersize=4.5, markeredgewidth=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(trial_conditions)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    ax.set_ylabel('Frequency of\nno change/other edits')
    ax.set_ylim(0,1.05)
    ax.set_xlim(-0.5, 3.5)
    # ax.set_title('Frequency of no change/other edits\ngiven mismatch condition', fontsize=10)
    ax.text(-0.5, 1.15, r"$\bf{Study}$"+' '+r"$\bf{"+"{}".format(study_index)+"}$", fontsize=12)

    # Plot the frequency of inferrer errors/target edits that resolve the agreement mismatch
    ax = fig.add_subplot(gs[1])
    np.random.seed(10)  # Set seed of the random jitter for reprocudible figures
    n_condition = len(trial_conditions)
    n_error = len(error_types)
    colors = [plt.cm.tab10(i) for i in range(n_condition)]
    # fontsize = 12
    
    for i in range(n_condition):
        xs = [e_idx+0.2*i for e_idx in range(n_error)]
        ys = np.mean(normalized_error_type_fdist_by_condition_all[i], axis=0)
        n_obs = len(normalized_error_type_fdist_by_condition_all[i])
        yerrs = 1.96*np.std(normalized_error_type_fdist_by_condition_all[i], axis=0)/np.sqrt(n_obs)
        ax.bar(xs, ys, width=0.12, label=trial_conditions[i], yerr=yerrs, color=colors[i])
        
        # Plot each data point with small jitter along x-axis
        for j in range(len(normalized_error_type_fdist_by_condition_all[i])):
            xjitter = np.random.uniform()*0.1-0.05
            xs = [e_idx+0.2*i+xjitter for e_idx in range(n_error)]
            ys = normalized_error_type_fdist_by_condition_all[i][j]
            ax.plot(xs, ys, 'ko', alpha=0.2, mfc='none', markersize=4.5, markeredgewidth=0.7)
                    
    ax.legend(ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    ax.set_xticks([x + 0.3 for x in range(n_error)])
    ax.set_xticklabels(xlabel_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='y', which='major')
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')    
    plt.show()


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


# Load priors estimated from the norming study
pi_list = json.load(open('data/estimated_prior_from_norming_task.json'))
pi_list = pi_list[:57]

# Visualize prior of each items as line plot
plot_priors(pi_list, savepath='fig/estimated_prior_prob.pdf')


EXP_NAMES = ['exp1', 'exp2']
study_index = ['I', 'II']

for exp_idx, EXP_NAME in enumerate(EXP_NAMES):
    data = json.load(open('data/{}_participant_data.json'.format(EXP_NAME)))

    trial_conditions = ['SSP', 'SPP', 'PSS', 'PPS']
    error_types = ['subj', 'verb', 'subj+local', 'verb+local']

    e2i = dict([(e, i) for i, e in enumerate(error_types)])
    cond2i = dict([(cond, i) for i, cond in enumerate(trial_conditions)])

    # error_type_count_by_subject_condition, error_type_count_by_item_condition = extract_e_type_count(
    #     data, trial_conditions, error_types, stimuli, group2idx, include_items=None, verbose=False)

    include_stimuli = []

    for i in range(len(stimuli)):
        # exclude a problematic item
        if i == 55:
            continue
        include_stimuli.append(i)

    print('Include item idex:', include_stimuli)
    include_stimuli = set(include_stimuli)

    error_type_count_by_subject_condition, error_type_count_by_item_condition = extract_e_type_count(
        data, trial_conditions, error_types, stimuli, group2idx, include_items=include_stimuli, verbose=False)

    print('Summary of cumulative counts:')
    display_subject_data(np.sum(error_type_count_by_subject_condition, axis=0), trial_conditions, error_types) 

    error_type_freq_by_subject_condition = np.zeros(error_type_count_by_subject_condition.shape)
    for k, subject_e_type_count in enumerate(error_type_count_by_subject_condition):
        e_type_freq = np.zeros((len(trial_conditions), len(error_types)))
        for i in range(len(trial_conditions)):
            if np.sum(subject_e_type_count[i]) >= 1:
            	e_type_freq[i] = normalize(subject_e_type_count[i])
            else:
            	print(subject_e_type_count)
        error_type_freq_by_subject_condition[k] = e_type_freq

    # print(error_type_freq_by_subject_condition)

    normalized_error_type_fdist_by_condition_all = [[], [], [], []]

    for normalized_fdist in error_type_freq_by_subject_condition:
        for i in range(4):
            if np.sum(normalized_fdist[i]) > 0:
                normalized_error_type_fdist_by_condition_all[i].append(normalized_fdist[i])

    savepath = 'fig/{}_e_type_freq.pdf'.format(EXP_NAME)
    xlabel_names = ['subject', 'verb', 'subject+local', 'verb+local']
        
    plot_inferred_error_frequency(
        normalized_error_type_fdist_by_condition_all, xlabel_names, figsize=(6.5, 2.4), seed=10,
        legend_ncol=1, savepath=savepath)

    other_edit_types = ['no_change', 'other'] if EXP_NAME == 'exp1' else ['skip','no_change','other']
    include_items = [idx for idx in range(55)] + [idx for idx in range(56, 57)]
    other_edit_type_count_by_subject_condition, other_edit_type_count_by_item_condition = extract_e_type_count(
        data, trial_conditions, other_edit_types, stimuli, group2idx, include_items=include_items)

    nonmismatch_edit_total = int(np.sum(other_edit_type_count_by_item_condition))
    mismatch_edit_total = int(np.sum(error_type_count_by_item_condition[include_items]))
    print('{:<4} ({:.3f}%) edits do not resolve agreement mismatch.\n{:<4} ({:.3f}%) edits resolve agreement mismatch.'.format(
        nonmismatch_edit_total, 
        100*nonmismatch_edit_total/(nonmismatch_edit_total+mismatch_edit_total), 
        mismatch_edit_total,
        100*mismatch_edit_total/(nonmismatch_edit_total+mismatch_edit_total)))

    other_edit_type_freq_by_subject_condition = np.zeros(other_edit_type_count_by_subject_condition.shape)
    for k, subject_e_type_count in enumerate(other_edit_type_count_by_subject_condition):
        e_type_freq = np.zeros((len(trial_conditions), len(other_edit_types)))
        for i in range(len(trial_conditions)):
            if np.sum(subject_e_type_count[i]) >= 1:
                e_type_freq[i] = normalize(subject_e_type_count[i])
        other_edit_type_freq_by_subject_condition[k] = e_type_freq

    normalized_other_edit_type_fdist_by_condition_all = [[], [], [], []]

    for normalized_fdist in other_edit_type_freq_by_subject_condition:
        for i in range(4):
            if np.sum(normalized_fdist[i]) > 0:
                normalized_other_edit_type_fdist_by_condition_all[i].append(normalized_fdist[i])


    # Visualize frequency of skip/no change/other edits given a trial condition
    np.random.seed(100)
    plt.figure(figsize=(2.8,2.4))
    ax = plt.gca()
    normalized_other_edit_freq_all = []
    for subject_idx, subject_other_edit_count in enumerate(other_edit_type_count_by_subject_condition):
        subject_other_edit_total_count = np.sum(subject_other_edit_count, axis=1)
        subject_target_edit_total_count = np.sum(error_type_count_by_subject_condition[subject_idx], axis=1)
        subject_total_count = subject_other_edit_total_count + subject_target_edit_total_count
        normalized_other_edit_freq_all.append(np.divide(subject_other_edit_total_count, subject_total_count))

    yerrs = 1.96*np.std(normalized_other_edit_freq_all, axis=0)/np.sqrt(len(normalized_other_edit_freq_all))
    plt.bar(np.arange(4), np.mean(normalized_other_edit_freq_all, axis=0), yerr=yerrs, 
            width=0.5, color='none', edgecolor='k', error_kw=dict(ecolor='r', capsize=4))
    for subject_other_edit_freq_given_condition in normalized_other_edit_freq_all:
        xjitter = np.random.uniform()*0.2-0.1
        plt.plot(np.arange(4)+xjitter, subject_other_edit_freq_given_condition, 'ko', mfc='none', alpha=0.3, markersize=4.5, markeredgewidth=0.8)
        
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(trial_conditions)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  

    # ax.set_ylabel('Frequency')
    ax.set_ylabel('Frequency of\nno change/other edits')
    ax.set_ylim(0,1.05)
    # ax.set_ylim(0,0.3)
    ax.set_xlim(-0.5, 3.5)
    # ax.set_title('Frequency of no change/other edits\ngiven mismatch condition', fontsize=10)
    plt.savefig('fig/{}_nonmismatch_edit_freq_given_condition.pdf'.format(EXP_NAME), bbox_inches='tight')
    plt.show()

    plot_error_correction_preference_panel(error_type_count_by_subject_condition, other_edit_type_count_by_subject_condition, 
        trial_conditions, error_types, study_index[exp_idx], figsize=(10, 2.5), savepath='fig/{}_panel.pdf'.format(EXP_NAME))



