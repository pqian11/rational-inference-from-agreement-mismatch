import numpy as np
import arviz
import json
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
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


def plot_model_human_comparison_across_two_studies(model_fit_all, observed_freqs_dict, model_names, pretty_model_names, savepath=None):
    n_cond = 4
    n_e_type = 4
    markersize = 5
    alpha = 0.8

    colors = [plt.cm.tab10(i) for i in range(4)]
    m_shapes = ['^', 'o', 's', 'x']
    study_names = ['I', 'II']

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6), sharey=True, sharex=True)

    for exp_idx, exp_name in enumerate(exp_names):
        observed_freqs = observed_freqs_dict[exp_name]
        for model_idx, model_name in enumerate(model_names):
            ax = axes[exp_idx, model_idx]

            summary = arviz.summary(model_fit_all[exp_name][model_name], var_names=['inferred_e_prob'], hdi_prob=0.95)
            print(summary)

            xs = []
            ys = []

            for i, index in enumerate(items):
                for j in range(n_cond):
                    observed_e_freq = normalize(observed_freqs[i, j])
                    inferred_e_prob = [summary['mean']['inferred_e_prob[{}, {}, {}]'.format(i,j,k)] for k in range(n_e_type)]

                    for e_type_idx in range(n_e_type):
                        ax.plot(inferred_e_prob[e_type_idx], observed_e_freq[e_type_idx], m_shapes[e_type_idx], 
                            alpha=alpha, mfc='none', color=colors[j], markersize=markersize)

                    xs += list(inferred_e_prob)
                    ys += list(observed_e_freq)
                    
            # ax.plot([0,1], [0,1], 'k--', alpha=0.5)
            ax.set_xlim(0,1.05)
            ax.set_ylim(0,1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if model_idx == 0:
                ax.set_ylabel(r"$\bf{Study}$"+' '+r"$\bf{"+"{}".format(study_names[exp_idx])+"}$"+'\nHuman correction preference', fontsize=12)

            mse = np.square(np.array(xs) - np.array(ys)).mean()
            pearsonr, _ = scipy.stats.pearsonr(xs, ys)
            rho, p_value = scipy.stats.spearmanr(xs, ys)
            sig_level_stars = get_significance_level_stars(p_value)

            print(scipy.stats.pearsonr(xs, ys))
            print(scipy.stats.spearmanr(xs, ys))
            print('mse', mse)

            pretty_model_name = pretty_model_names[model_idx]

            if exp_idx == 0:
                ax.set_title(pretty_model_name)
            text_patch = ax.text(0.56, 0.03, r'MSE$=${:.3f}'.format(mse)+'\n'+r'$\rho=${:.3f}{}'.format(rho, sig_level_stars), transform=ax.transAxes)   
            text_patch.set_bbox(dict(facecolor='white', alpha=0.25, edgecolor='none'))

    fig.text(0.5, 0.035, 'Model correction preference', ha='center', fontsize=12)

    ax = fig.add_subplot(111)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')    
    e_type_legend_elements = []
    for e_type_idx, e_type in enumerate(['subject', 'verb', 'subject+local', 'verb+local']):
        e_type_legend_elements.append(Line2D([0], [0], marker=m_shapes[e_type_idx], mfc='none', label=e_type, linestyle='None', markersize=markersize, color='gray'))
    legend = plt.legend(handles=e_type_legend_elements, loc='upper left', ncol=4, bbox_to_anchor=(0.45, -0.12), title='Type of correction/inferred error')
    ax.add_artist(legend)
    condition_legend_elements = []
    for cond_idx, cond in enumerate(['SSP', 'SPP', 'PSS', 'PPS']):
        patch = mpatches.Patch(color=colors[cond_idx], label=cond)
        condition_legend_elements.append(patch)
    legend = plt.legend(handles=condition_legend_elements, ncol=4, loc='upper left', bbox_to_anchor=(0.05, -0.12), title='Condition')
    ax.add_artist(legend)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()   


def plot_model_human_comparison_with_combined_data(model_fit_all, observed_freqs, model_names, pretty_model_names, savepath=None):
    n_cond = 4
    n_e_type = 4
    markersize = 5
    alpha = 0.65

    colors = [plt.cm.tab10(i) for i in range(4)]
    m_shapes = ['^', 'o', 's', 'x']
    study_names = ['I', 'II']

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3), sharey=True, sharex=True)

    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]

        # ax.plot([0,1], [0,1], 'k', ls='--', alpha=0.45)

        summary = arviz.summary(model_fit_all[model_name], var_names=['inferred_e_prob'], hdi_prob=0.95)
        print(summary)

        xs = []
        ys = []

        for i, index in enumerate(items):
            for j in range(n_cond):
                observed_e_freq = normalize(observed_freqs[i, j])
                inferred_e_prob = [summary['mean']['inferred_e_prob[{}, {}, {}]'.format(i,j,k)] for k in range(n_e_type)]

                for e_type_idx in range(n_e_type):
                    ax.plot(inferred_e_prob[e_type_idx], observed_e_freq[e_type_idx], m_shapes[e_type_idx], 
                        alpha=alpha, mfc='none', color=colors[j], markersize=markersize)

                xs += list(inferred_e_prob)
                ys += list(observed_e_freq)
                
        
        ax.set_xlim(0,1.05)
        ax.set_ylim(0,1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if model_idx == 0:
            ax.set_ylabel('Human correction preference', fontsize=12)

        mse = np.square(np.array(xs) - np.array(ys)).mean()
        pearsonr, _ = scipy.stats.pearsonr(xs, ys)
        rho, p_value = scipy.stats.spearmanr(xs, ys)
        r2 = r2_score(ys, xs)
        sig_level_stars = get_significance_level_stars(p_value)

        print(scipy.stats.pearsonr(xs, ys))
        print(scipy.stats.spearmanr(xs, ys))
        print('mse', mse)
        print('r2',r2)

        pretty_model_name = pretty_model_names[model_idx]

        # if exp_idx == 0:
        ax.set_title(pretty_model_name)
        text_patch = ax.text(0.56, 0.03, r'MSE$=${:.3f}'.format(mse)+'\n'+r'$\rho=${:.3f}{}'.format(rho, sig_level_stars), transform=ax.transAxes)   
        text_patch.set_bbox(dict(facecolor='white', alpha=0.25, edgecolor='none'))

    fig.text(0.5, -0.03, 'Model correction preference', ha='center', fontsize=12)

    legends = []
    ax = fig.add_subplot(111)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')    
    e_type_legend_elements = []
    for e_type_idx, e_type in enumerate(['subject', 'verb', 'subject+local', 'verb+local']):
        e_type_legend_elements.append(Line2D([0], [0], marker=m_shapes[e_type_idx], mfc='none', label=e_type, linestyle='None', markersize=markersize, color='gray'))
    legend = plt.legend(handles=e_type_legend_elements, loc='upper left', ncol=4, bbox_to_anchor=(0.45, -0.2), title='Type of correction/inferred error')
    ax.add_artist(legend)

    legends.append(legend)

    condition_legend_elements = []
    for cond_idx, cond in enumerate(['SSP', 'SPP', 'PSS', 'PPS']):
        patch = mpatches.Patch(color=colors[cond_idx], label=cond)
        condition_legend_elements.append(patch)
    legend = plt.legend(handles=condition_legend_elements, ncol=4, loc='upper left', bbox_to_anchor=(0.05, -0.2), title='Condition')
    ax.add_artist(legend)

    legends.append(legend)


    if savepath:
        plt.savefig(savepath, bbox_extra_artists=legends, bbox_inches='tight')
    plt.show()   


model_names = ['bayesian_model_prior-only_temperature_with_random_effect', 
                'bayesian_model_lh-only_with_random_effect', 
                'bayesian_model_context-general_lh_temperature_with_random_effect', 
                'bayesian_model_full_temperature_with_random_effect']

pretty_model_names = ['Prior only', 'Likelihood only', 'Context-insensitive', 'Full model']

e_types = ['subj', 'verb', 'subj+local', 'verb+local']
e_type2idx = dict([(e_type, i) for i, e_type in enumerate(e_types)])
items = [k for k in range(57) if k != 55] # Exclude a problematic item
item2idx = dict([(item, i) for i, item in enumerate(items)])
conditions = ['SSP', 'SPP', 'PSS', 'PPS']
condition2idx = dict([(condition, i) for i, condition in enumerate(conditions)])

exp_names = ['exp1', 'exp2']

# Load observed frequency count of edit/error types as well as model fits for Study I and II separately
observed_freqs_dict = {}
model_fit_all = {}

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
    observed_freqs_dict[exp_name] = observed_all

    model_fit_all[exp_name] = {}
    for model_name in model_names:
        print('Load model fit of {} for {}...'.format(model_name, exp_name))
        model_fit_path = "stan_model/model_fit/{}_{}_model_fit.nc".format(exp_name, model_name)
        model_fit_all[exp_name][model_name] = arviz.from_netcdf(model_fit_path)

savepath = 'fig/human_model_comparison_all.pdf'
plot_model_human_comparison_across_two_studies(model_fit_all, observed_freqs_dict, model_names, pretty_model_names, savepath=savepath)


# Load combined observed frequency count of edit/error types as well as model fits on combined data
observed_all = np.zeros((len(items), len(conditions), len(e_types)))
for exp_name in exp_names:
    # Load data frame
    df = pd.read_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name), index_col=0)

    # Add edit type count
    for row_idx, row in df.iterrows():
        item = row['item']
        trial_condition = row['trial_condition']
        e_type = row['e_type']
        observed_all[item2idx[item], condition2idx[trial_condition], e_type2idx[e_type]] += 1

for model_name in model_names:
    print('Load model fit of {} for combined data...'.format(model_name))
    model_fit_path = "stan_model/model_fit/exp_combined_{}_model_fit.nc".format(model_name)
    model_fit_all[model_name] = arviz.from_netcdf(model_fit_path)

savepath = 'fig/human_model_comparison_with_combined_data.pdf'
plot_model_human_comparison_with_combined_data(model_fit_all, observed_all, model_names, pretty_model_names, savepath=savepath)