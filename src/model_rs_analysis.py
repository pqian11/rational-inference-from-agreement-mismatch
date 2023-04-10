import numpy as np
import arviz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
rcParams['font.family'] = 'Arial'


def plot_inferred_params_posterior(summaries, study_names=None, figsize=(5.5, 4), savepath=None):
    condition_names = ['SS_', 'SP_', 'PS_', 'PP_']
    lh_params_posterior_ci_all = []
    n_exp = len(summaries)
    if study_names:
        assert(len(summaries) == len(study_names))

    for i in range(n_exp):
        lh_params_posterior_ci = []
        for k in range(4):
            lh_params_posterior_ci.append(
                [(
                    summaries[i]['mean']['error_prob[{}, {}]'.format(k,j)], 
                    summaries[i]['mean']['error_prob[{}, {}]'.format(k,j)] - summaries[i]['hdi_2.5%']['error_prob[{}, {}]'.format(k,j)], 
                    summaries[i]['hdi_97.5%']['error_prob[{}, {}]'.format(k,j)] - summaries[i]['mean']['error_prob[{}, {}]'.format(k,j)]
                ) for j in range(4)])
        lh_params_posterior_ci_all.append(lh_params_posterior_ci)
        
    # Vertical layout
    fig, axes = plt.subplots(1, len(summaries), figsize=figsize)
    for ax_idx in range(n_exp):
        ax = axes[ax_idx]
        lh_params_posterior_ci = lh_params_posterior_ci_all[ax_idx]
        for k in range(4):
            ax.errorbar([item[0] for item in lh_params_posterior_ci[k]], np.arange(16)[::-1][np.arange(k*4, (k+1)*4)],
                        marker='o', mfc='white', ms=4, ls='',
                        xerr=[[item[1] for item in lh_params_posterior_ci[k]], 
                              [item[2] for item in lh_params_posterior_ci[k]]], 
                        label=condition_names[k])
            ax.set_xlim(0, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax.set_yticks(np.arange(16))
        ax.set_xlabel('Error Frequency')

        if study_names:
            ax.set_title(study_names[ax_idx])
        
        if ax_idx == 0:
            yticklabels = []
            e_types = ['subject', 'verb', 'subject+local', 'verb+local']
            for k in range(4):
                for j in range(4):
                    yticklabels.append('{} | {}'.format(e_types[j], condition_names[k]))

            ax.set_yticklabels(yticklabels[::-1])
        else:
            ax.set_yticklabels([])
            
    condition_legend_elements = []
    for i, condition in enumerate(condition_names):
        line = mlines.Line2D([], [], color=plt.cm.tab10(i), marker='o', label=condition, mfc='white')
        condition_legend_elements.append(line)
    legend = fig.legend(handles=condition_legend_elements, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


model_names = ['bayesian_model_prior-only_temperature_with_random_effect',
                'bayesian_model_lh-only_with_random_effect',
                'bayesian_model_context-general_lh_temperature_with_random_effect',
                'bayesian_model_full_temperature_with_random_effect'
                ]

exp_names = ['exp1', 'exp2', 'exp_combined']

for model_name in model_names:
    model_summary_all = []
    for exp_name in exp_names:
        model_fit_path = "stan_model/model_fit/{}_{}_model_fit.nc".format(exp_name, model_name)
        model_fit = arviz.from_netcdf(model_fit_path)

        if model_name.startswith('bayesian_model_full_temperature'):
            var_name_list = ['error_prob', 'delta_SP_SS', 'delta_PS_PP', 'delta_SP_PS', 'a', 'sigma_item', 'sigma_subject', 'log_lh']
        elif model_name.startswith('bayesian_model_prior-only_temperature'):
            var_name_list = ['a', 'sigma_item', 'sigma_subject', 'log_lh']
        elif model_name.startswith('bayesian_model_lh-only'):
            var_name_list = ['error_prob', 'sigma_item', 'sigma_subject', 'log_lh']
        else:
            var_name_list = ['error_prob', 'a', 'sigma_item', 'sigma_subject', 'log_lh']

        summary = arviz.summary(model_fit, var_names=var_name_list, hdi_prob=0.95)

        print('Summary of {} fitted on data from {}:'.format(model_name, exp_name))
        print(summary)
        print()

        model_summary_all.append(summary)

    # Plot inferred error likelihood parameters for Study I and Study II separately.
    if not model_name.startswith('bayesian_model_prior-only_temperature'):
        savepath = 'fig/estimated_lh_parmas_{}.pdf'.format(model_name)
        plot_inferred_params_posterior(model_summary_all[:2], study_names=['Study I', 'Study II'], figsize=(7, 3.5), savepath=None)
