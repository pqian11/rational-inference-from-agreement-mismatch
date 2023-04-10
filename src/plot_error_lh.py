import numpy as np
import arviz
import json
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
rcParams['font.family'] = 'Arial'


def pprint_error_count_by_condition(error_freq_count_by_condition):
    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format('Condition', *e_types))
    for cond in conditions:
        print('{:<12} {:<12} {:<12} {:<12} {:<12}'.format(cond, *[error_freq_count_by_condition[cond][e_type] for e_type in e_types] ))  
        
        
def get_error_freq_count_summary(error_freq_count_by_condition):
    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    error_freq_summary = {}
    for cond in conditions:
        error_freq_summary[cond] = {}
        total_count = np.sum([error_freq_count_by_condition[cond][e_type] for e_type in e_types])
#         print(total_count)
        assert total_count > 0
        for e_type in e_types:
            e_type_count = error_freq_count_by_condition[cond][e_type]
            ci_low, ci_up = proportion_confint(e_type_count, total_count, method='normal')
            e_freq = e_type_count/total_count
            error_freq_summary[cond][e_type] = {}
            error_freq_summary[cond][e_type]['mean'] = e_freq
            error_freq_summary[cond][e_type]['ci_2.5%'] = ci_low
            error_freq_summary[cond][e_type]['ci_97.5%'] = ci_up
    return error_freq_summary
            

def plot_lh_params_all(summaries, figsize=(10, 2), subplot_titles=None, savepath=None):
    condition_names = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    lh_params_ci_list = []
    for summary in summaries:
        lh_params_ci = []
        for k, cond in enumerate(condition_names):
            lh_params_ci.append(
                [(
                    summary[cond][e_type]['mean'], 
                    np.absolute(summary[cond][e_type]['mean'] - summary[cond][e_type]['ci_2.5%']), 
                    np.absolute(summary[cond][e_type]['ci_97.5%'] - summary[cond][e_type]['mean']),
                ) for e_type in e_types])   
        lh_params_ci_list.append(lh_params_ci)
        
    # Generate ticklabels
    ticklabels = []
    for k in range(4):
        ticklabels += ['{} | {}'.format(e_type,condition_names[k]) for e_type in e_types]        
        
    if subplot_titles:
        assert len(subplot_titles) == len(summaries)
        
    fig, axes = plt.subplots(1, len(summaries), sharey=True, figsize=figsize)
    for ax_idx, ax in enumerate(axes):
        lh_params_ci = lh_params_ci_list[ax_idx]
        for k in range(4):
            ax.errorbar([item[0] for item in lh_params_ci[k]], np.arange(16)[::-1][np.arange(k*4, (k+1)*4)],
                        marker='o', mfc='white', ms=4, ls='',
                        xerr=[[item[1] for item in lh_params_ci[k]], 
                                           [item[2] for item in lh_params_ci[k]]], label=condition_names[k]
                                      )
            ax.set_xlim(0, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        ax.set_yticks(np.arange(16))
        if ax_idx == 0:
            ax.set_yticklabels(ticklabels[::-1])
        ax.set_xlabel('Error Frequency')
        if subplot_titles:
            ax.set_title(subplot_titles[ax_idx])

        if ax_idx == 0:
            ax.annotate('Comprehension', xy=(1, 1.28), xytext=(1, 1.28), xycoords='axes fraction', 
            ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white', lw=0),
            arrowprops=dict(arrowstyle='-[, widthB=7, lengthB=1'))

        if ax_idx == 3:
            ax.annotate('Production', xy=(0.5, 1.28), xytext=(0.5, 1.28), xycoords='axes fraction', 
            ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white', lw=0),
            arrowprops=dict(arrowstyle='-[, widthB=15, lengthB=1'))

    condition_legend_elements = []
    for i, condition in enumerate(condition_names):
        line = mlines.Line2D([], [], color=plt.cm.tab10(i), marker='o', label=condition, mfc='white')
        condition_legend_elements.append(line)
    # legend = fig.legend(handles=condition_legend_elements, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.02))
    legend = fig.legend(handles=condition_legend_elements, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.15))
            
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return


def plot_lh_params_for_model_on_combined_data(summaries, figsize=(4, 2), subplot_titles=None, savepath=None):
    condition_names = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    lh_params_ci_list = []
    for summary in summaries:
        lh_params_ci = []
        for k, cond in enumerate(condition_names):
            lh_params_ci.append(
                [(
                    summary[cond][e_type]['mean'], 
                    summary[cond][e_type]['mean'] - summary[cond][e_type]['ci_2.5%'], 
                    summary[cond][e_type]['ci_97.5%'] - summary[cond][e_type]['mean'],
                ) for e_type in e_types])   
        lh_params_ci_list.append(lh_params_ci)
        
    # Generate ticklabels
    ticklabels = []
    for k in range(4):
        ticklabels += ['{} | {}'.format(e_type,condition_names[k]) for e_type in e_types]        
        
    if subplot_titles:
        assert len(subplot_titles) == len(summaries)
        
    fig, axes = plt.subplots(1, len(summaries), sharey=True, figsize=figsize)
    for ax_idx, ax in enumerate(axes):
        lh_params_ci = lh_params_ci_list[ax_idx]
        for k in range(4):
            ax.errorbar([item[0] for item in lh_params_ci[k]], np.arange(16)[::-1][np.arange(k*4, (k+1)*4)],
                        marker='o', mfc='white', ms=4, ls='',
                        xerr=[[item[1] for item in lh_params_ci[k]], 
                                           [item[2] for item in lh_params_ci[k]]], label=condition_names[k]
                                      )
            ax.set_xlim(0, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        ax.set_yticks(np.arange(16))
        if ax_idx == 0:
            ax.set_yticklabels(ticklabels[::-1])
        ax.set_xlabel('Error Frequency')
        if subplot_titles:
            ax.set_title(subplot_titles[ax_idx])

    condition_legend_elements = []
    for i, condition in enumerate(condition_names):
        line = mlines.Line2D([], [], color=plt.cm.tab10(i), marker='o', label=condition, mfc='white')
        condition_legend_elements.append(line)
    legend = fig.legend(handles=condition_legend_elements, ncol=1, loc='center left', bbox_to_anchor=(0.92, 0.5))
            
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return


lh_params_summaries = {}

# Summarize Ryskin et al Exp1 data
e_dict_ryskin_exp1a = json.load(open('data/ryskin_et_al_exp1a_error_count.json'))
lh_params_summaries['ryskin_et_al_exp1a'] = get_error_freq_count_summary(e_dict_ryskin_exp1a)

# Summarize Kandel et al Exp1 data
e_dict_kandel_exp1= json.load(open('data/kandel_et_al_exp1_error_count.json'))
lh_params_summaries['kandel_et_al_exp1'] = get_error_freq_count_summary(e_dict_kandel_exp1)

# Summarize Kandel et al Exp2 data
e_dict_kandel_exp2= json.load(open('data/kandel_et_al_exp2_error_count.json'))
lh_params_summaries['kandel_et_al_exp2'] = get_error_freq_count_summary(e_dict_kandel_exp2)


# Summarize estimated error likelihood parameters from the full Bayesian model
model_name = 'bayesian_model_full_temperature_with_random_effect'
exp_names = ['exp1', 'exp2']

model_fit_all = []

for exp_name in exp_names:
    model_fit_path = "stan_model/model_fit/{}_{}_model_fit.nc".format(exp_name, model_name)
    model_fit = arviz.from_netcdf(model_fit_path)
    if model_name == 'bayesian_model_full_temperature':
        summary = arviz.summary(model_fit, var_names=['error_prob', 'delta_SP_SS', 'delta_PS_PP', 'delta_SP_PS', 'a', 'log_lh'], hdi_prob=0.95)
    else:
        summary = arviz.summary(model_fit, var_names=['error_prob', 'log_lh'], hdi_prob=0.95)

    print('Summary of {} fitted on data from {}:'.format(model_name, exp_name))
    print(summary)
    print()

    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject', 'verb', 'subject+local', 'verb+local']
    lh_params_posterior_summary = {}
    for k, cond in enumerate(conditions):
        lh_params_posterior_summary[cond] = {}
        for j, e_type in enumerate(e_types):
            lh_params_posterior_summary[cond][e_type] = {}
            lh_params_posterior_summary[cond][e_type]['mean'] = summary['mean']['error_prob[{}, {}]'.format(k,j)]
            lh_params_posterior_summary[cond][e_type]['ci_2.5%'] = summary['hdi_2.5%']['error_prob[{}, {}]'.format(k,j)]
            lh_params_posterior_summary[cond][e_type]['ci_97.5%'] = summary['hdi_97.5%']['error_prob[{}, {}]'.format(k,j)]           

    lh_params_summaries[exp_name] = lh_params_posterior_summary

    model_fit_all.append(model_fit)


summary_names = ['exp1', 'exp2', 'ryskin_et_al_exp1a', 'kandel_et_al_exp1', 'kandel_et_al_exp2']
summaries = [lh_params_summaries[summary_name] for summary_name in summary_names]
subplot_titles = ['Study I',
                  'Study II',
                  'Ryskin et al. (2021)\nExperiment 1a', 
                  'Kandel et al. (2022)\nExperiment 1', 
                  'Kandel et al. (2022)\nExperiment 2']
savepath = 'fig/error_lh_comparison.pdf'
plot_lh_params_all(summaries, figsize=(12,3), subplot_titles=subplot_titles, savepath=savepath)


# Plot results from full model fitted on combined data from Study I&II as well as meta-analysis results
# of language production studies
model_name = 'bayesian_model_full_temperature_with_random_effect'
model_fit_path = "stan_model/model_fit/exp_combined_{}_model_fit.nc".format(model_name)
model_fit = arviz.from_netcdf(model_fit_path)
summary = arviz.summary(model_fit, var_names=['error_prob', 'log_lh'], hdi_prob=0.95)

print('Summary of {} fitted on data from combined data.'.format(model_name))
print(summary)
print()

lh_params_posterior_summary = {}
for k, cond in enumerate(conditions):
    lh_params_posterior_summary[cond] = {}
    for j, e_type in enumerate(e_types):
        lh_params_posterior_summary[cond][e_type] = {}
        lh_params_posterior_summary[cond][e_type]['mean'] = summary['mean']['error_prob[{}, {}]'.format(k,j)]
        lh_params_posterior_summary[cond][e_type]['ci_2.5%'] = summary['hdi_2.5%']['error_prob[{}, {}]'.format(k,j)]
        lh_params_posterior_summary[cond][e_type]['ci_97.5%'] = summary['hdi_97.5%']['error_prob[{}, {}]'.format(k,j)]           

lh_params_summaries['exp_combined'] = lh_params_posterior_summary

# Load meta-analysis results
model_fit_path = "stan_model/model_fit/meta_analysis_model_fit.nc"
model_fit = arviz.from_netcdf(model_fit_path)
summary = arviz.summary(model_fit, var_names=['mu'], hdi_prob=0.95)
print('Summary of meta-analysis model fitted on data from language production studies.')
print(summary)
print()

lh_params_posterior_summary = {}
for k, cond in enumerate(conditions):
    lh_params_posterior_summary[cond] = {}
    for j, e_type in enumerate(e_types):
        lh_params_posterior_summary[cond][e_type] = {}
        lh_params_posterior_summary[cond][e_type]['mean'] = summary['mean']['mu[{}, {}]'.format(k,j)]
        lh_params_posterior_summary[cond][e_type]['ci_2.5%'] = summary['hdi_2.5%']['mu[{}, {}]'.format(k,j)]
        lh_params_posterior_summary[cond][e_type]['ci_97.5%'] = summary['hdi_97.5%']['mu[{}, {}]'.format(k,j)]           

lh_params_summaries['production'] = lh_params_posterior_summary

summary_names = ['exp_combined', 'production']
summaries = [lh_params_summaries[summary_name] for summary_name in summary_names]
subplot_titles = ['Error correction', 'Production']
savepath = 'fig/error_lh_comparison_combined.pdf'
plot_lh_params_for_model_on_combined_data(summaries, figsize=(7,3), subplot_titles=subplot_titles, savepath=savepath)
