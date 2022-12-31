import numpy as np
import json
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
rcParams['font.family'] = 'Arial'


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


def get_significance_level_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def plot_verb_correction_freq_against_prior_panel(obs_e_freqs, prior_probs, items, study_index, savepath=None, figsize=None):
   # Plot conditions as separate graph
    if figsize is None:
        figsize = (11, 5)

    condition_names = ['SSP', 'SPP', 'PSS', 'PPS']
    n_cond = len(condition_names)

    colors = [plt.cm.tab10(i) for i in range(4)]

    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs_all = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.15)
    gs = gs_all[1].subgridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                        wspace=0.3, hspace=0.3)

    ax = fig.add_subplot(gs_all[0])

    cond2data = {}
    for k in range(n_cond):
        cond2data[k] = {'xs':[], 'ys':[]}

    xs = []
    ys = []

    for n, index in enumerate(items):
        for k in range(n_cond):
            observed_e_freq = obs_e_freqs[n][k]
            prior_prob = prior_probs[n][k]
             
            cond2data[k]['xs'].append(prior_prob[1])
            cond2data[k]['ys'].append(observed_e_freq[1])

            ax.plot(prior_prob[1], observed_e_freq[1], 'o', alpha=0.9, mfc='none', color=colors[k], markersize=6)

            xs.append(prior_prob[1])
            ys.append(observed_e_freq[1])

    cond_legend_elements = []
    for k in range(n_cond):
        cond_legend_elements.append(Line2D([0], [0], marker='o', label=condition_names[k], linestyle='None', markersize=6, mfc='none', color=colors[k]))
    legend = plt.legend(handles=cond_legend_elements, ncol=1, loc='center', bbox_to_anchor=(0.9, 0.25), title='Condition')
    ax.add_artist(legend)  
  

    ax1= fig.add_subplot(gs[0,0])
    ax2= fig.add_subplot(gs[0,1])
    ax3= fig.add_subplot(gs[1,0])
    ax4= fig.add_subplot(gs[1,1])

    axes = [ax1, ax2, ax3, ax4]

    for j in range(n_cond):
        for k in range(n_cond):
            if k == j:
                axes[k].plot(cond2data[j]['xs'], cond2data[j]['ys'], 'o', alpha=0.9, mfc='none', color=colors[j], markersize=5, zorder=10)
            else:
                axes[k].plot(cond2data[j]['xs'], cond2data[j]['ys'], 'o', alpha=0.06, mfc='none', color='k', markersize=5, zorder=0)     

        rho, p_value = scipy.stats.spearmanr(cond2data[j]['xs'], cond2data[j]['ys'])
        sig_level_stars = get_significance_level_stars(p_value)
        axes[j].text(0.5, 0.08, r'$\rho_{{{}}}=${:.3f}{}'.format(condition_names[j], rho, sig_level_stars), transform=axes[j].transAxes)  
        axes[j].set_title(condition_names[j], fontsize=10) 
        print('Study {} {}: rho={:.3f}, p={:.5f}'.format(study_index, condition_names[j], rho, p_value))

    rho, p_value = scipy.stats.spearmanr(xs, ys)
    sig_level_stars = get_significance_level_stars(p_value)
    ax.text(0.77, 0.05, r'$\rho_{{all}}=${:.3f}{}'.format(rho, sig_level_stars), transform=ax.transAxes)
    print('Study {} all: rho={:.3f}, p={:.5f}'.format(study_index, rho, p_value))

    for k in range(n_cond):
        axes[k].spines['top'].set_visible(False)
        axes[k].spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.02, 'Prior of the observed subject phrase', ha='center', fontsize=11)
    fig.text(0.08, 0.5, 'Verb correction frequency', va='center', rotation='vertical', fontsize=11)

    fig.text(0.5, 0.95, r"$\bf{Study}$"+' '+r"$\bf{"+"{}".format(study_index)+"}$", ha='center', fontsize=14)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show() 
    return


def plot_verb_correction_freq_against_prior_all_conditions(obs_e_freqs, prior_probs, items, study_index, figsize=(4,4), savepath=None):
   # Plot all conditions in one figure
    condition_names = ['SSP', 'SPP', 'PSS', 'PPS']
    n_cond = len(condition_names)

    colors = [plt.cm.tab10(i) for i in range(4)]

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    cond2data = {}
    for k in range(n_cond):
        cond2data[k] = {'xs':[], 'ys':[]}

    xs = []
    ys = []

    for n, index in enumerate(items):
        for k in range(n_cond):
            observed_e_freq = obs_e_freqs[n][k]
            prior_prob = prior_probs[n][k]
                    
            cond2data[k]['xs'].append(prior_prob[1])
            cond2data[k]['ys'].append(observed_e_freq[1])

            ax.plot(prior_prob[1], observed_e_freq[1], 'o', alpha=0.9, mfc='none', color=colors[k], markersize=6)

            xs.append(prior_prob[1])
            ys.append(observed_e_freq[1])

    cond_legend_elements = []
    for k in range(n_cond):
        cond_legend_elements.append(Line2D([0], [0], marker='o', label=condition_names[k], linestyle='None', markersize=6, mfc='none', color=colors[k]))
    legend = plt.legend(handles=cond_legend_elements, ncol=1, loc='center', bbox_to_anchor=(0.88, 0.25))
    ax.add_artist(legend)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(r"$\bf{Study}$"+' '+r"$\bf{"+"{}".format(study_index)+"}$")

    rho, p_value = scipy.stats.spearmanr(xs, ys)
    sig_level_stars = get_significance_level_stars(p_value)
    ax.text(0.7, 0.05, r'$\rho_{{all}}=${:.3f}{}'.format(rho, sig_level_stars), transform=ax.transAxes)

    plt.xlabel('Prior of the observed subject phrase')
    plt.ylabel('Verb correction frequency')

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return


# Load estimated prior
pi_list = json.load(open('data/estimated_prior_from_norming_task.json'))
pi_list = np.array(pi_list)

exp_names = ['exp1', 'exp2']
study_index = ['I', 'II']

e_types = ['subj', 'verb', 'subj+local', 'verb+local']
e_type2idx = dict([(e_type, i) for i, e_type in enumerate(e_types)])
items = [k for k in range(57) if k != 55] # Exclude a problematic item
item2idx = dict([(item, i) for i, item in enumerate(items)])
conditions = ['SSP', 'SPP', 'PSS', 'PPS']
condition2idx = dict([(condition, i) for i, condition in enumerate(conditions)])


# Load observed frequency count of edit/error types
observed_freqs = {}
for exp_name in exp_names:
    # Load data frame
    df = pd.read_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name), index_col=0)

    # Get edit type counts for each condition of every item
    observed_all = np.zeros((len(items), len(conditions), len(e_types)))
    for row_idx, row in df.iterrows():
        item = row['item']
        trial_condition = row['trial_condition']
        e_type = row['e_type']
        observed_all[item2idx[item], condition2idx[trial_condition], e_type2idx[e_type]] += 1
    observed_freqs[exp_name] = observed_all


for exp_idx, exp_name in enumerate(exp_names):
    observed_all = observed_freqs[exp_name]

    obs_e_freqs = []
    prior_probs = []

    for n, index in enumerate(items):
        obs_e_freqs.append([])
        prior_probs.append([])
        for k in range(len(conditions)):
            observed_e_freq = normalize(observed_all[n, k])
            
            if k == 0:
                prior_prob = [pi_list[index][2], pi_list[index][k], pi_list[index][3], pi_list[index][1]]
            elif k == 1:
                prior_prob = [pi_list[index][3], pi_list[index][k], pi_list[index][2], pi_list[index][0]]
            elif k == 2:
                prior_prob = [pi_list[index][0], pi_list[index][k], pi_list[index][1], pi_list[index][3]]    
            else:
                prior_prob = [pi_list[index][1], pi_list[index][k], pi_list[index][0], pi_list[index][2]]

            obs_e_freqs[n].append(observed_e_freq)
            prior_probs[n].append(prior_prob)

    plot_verb_correction_freq_against_prior_panel(obs_e_freqs, prior_probs, items, study_index[exp_idx], savepath='fig/{}_verb_correction_freq_prior_panel.pdf'.format(exp_name))
    plot_verb_correction_freq_against_prior_all_conditions(obs_e_freqs, prior_probs, items, study_index[exp_idx], savepath='fig/{}_verb_correction_freq_prior.pdf'.format(exp_name))



