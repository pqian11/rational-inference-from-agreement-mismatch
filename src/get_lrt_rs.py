import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import json
import os.path
import scipy.stats
from scipy.stats.distributions import chi2


def print_header(line):
    print(line)
    print('-'*len(line))


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


def get_log_lh(data, inferred_e_prob, sigma):
    """
    Given the observed edit type distribution, inferred edit type distribution,
    and standard deviation (sigma) of the noise term, calculate log likelihood 
    of the generative regression model (including the constant of normal distribution)
    """
    log_lh = 0
    N, K, J = data.shape
    for n in range(N):
        for k in range(K):
            for j in range(J):
                log_lh += scipy.stats.norm.logpdf(data[n, k, j], loc=inferred_e_prob[n, k, j], scale=sigma)
    return log_lh


def get_AIC(k, log_L):
    """
    k: the number of estimated parameters in the model
    log_L: the maximum value of the likelihood for the model
    """
    return 2*k - 2*log_L


def get_BIC(k, n, log_L):
    """
    k: the number of estimated parameters in the model
    n: the number of observations
    log_L: the maximum value of the likelihood for the model
    """
    return k*np.log(n) - 2*log_L 


def build_model(path, pkl_file=None, do_compile=True):
    if do_compile:
        sm = pystan.StanModel(file=path)
        if pkl_file is not None:
            with open(pkl_file, 'wb') as f:
                pickle.dump(sm, f)
    else:
        if os.path.isfile(pkl_file):
            sm = pickle.load(open(pkl_file, 'rb'))
        else:
            raise FileNotFoundError
    return sm


def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))


# Load estimated prior
pi_list = json.load(open('data/estimated_prior_from_norming_task.json'))
pi_list = np.array(pi_list)

model2name = {"context-general_lh":"bayesian_model_context-general_lh_temperature_aggregate",
                "full":"bayesian_model_full_temperature_aggregate"}
model_tags = ["context-general_lh", "full"]

models = {}
for model_tag in model_tags:
    model_name = model2name[model_tag]
    model_path = 'stan_model/{}.stan'.format(model_name)
    sm = CmdStanModel(stan_file=model_path)
    models[model_tag] = sm


# Maximum likelihood estimation for all the models using data from the two error-correction studies
exp_names = ['exp1', 'exp2']
mle_op_all = {}

e_types = ['subj', 'verb', 'subj+local', 'verb+local']
e_type2index = dict([(e_type, i) for i, e_type in enumerate(e_types)])
items = [k for k in range(57) if k != 55] # Exclude a problematic item
item2index = dict([(item, i) for i, item in enumerate(items)])
conditions = ['SSP', 'SPP', 'PSS', 'PPS']
condition2index = dict([(condition, i) for i, condition in enumerate(conditions)])

for exp_name in exp_names:
    mle_op_all[exp_name] = {}

    # Load data frame
    df = pd.read_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name), index_col=0)

    # Get edit type counts for each condition of every item
    observed_all = np.zeros((len(items), len(conditions), len(e_types)))
    for row_idx, row in df.iterrows():
        item = row['item']
        trial_condition = row['trial_condition']
        e_type = row['e_type']
        observed_all[item2index[item], condition2index[trial_condition], e_type2index[e_type]] += 1

    normalized_observed_all = []

    for n in range(len(observed_all)):
        normalized_observed_all.append([])
        for k in range(4):
            normalized_observed_all[n].append(normalize(observed_all[n][k]))

    normalized_observed_all = np.array(normalized_observed_all)
    # print(normalized_observed_all)

    data = {
      "N": 56,
      "K": 4,
      "J": 4,
      "obs_e_prob": normalized_observed_all,
      "prior": np.array(pi_list[items])
    }

    for k, model_tag in enumerate(model_tags):
        sm = models[model_tag]
        mle_op = sm.optimize(data=data, seed=1)

        print('log likelihood evaluated at the MLE of the parameters for {} is:\n'.format(model2name[model_tag]), 
              mle_op.optimized_params_dict['log_lh'], get_log_lh(data['obs_e_prob'], mle_op.stan_variables()['inferred_e_prob'], mle_op.optimized_params_dict['sigma']))

        mle_op_all[exp_name][model_tag] = mle_op.optimized_params_dict
        print()


# Maximum likelihood estimation for all the models using combined data
mle_op_all['combined'] = {}
observed_all = np.zeros((len(items), len(conditions), len(e_types)))
for exp_name in exp_names:
    # Load data frame
    df = pd.read_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name), index_col=0)

    # Get edit type counts for each condition of every item
    for row_idx, row in df.iterrows():
        item = row['item']
        trial_condition = row['trial_condition']
        e_type = row['e_type']
        observed_all[item2index[item], condition2index[trial_condition], e_type2index[e_type]] += 1

normalized_observed_all = []

for n in range(len(observed_all)):
    normalized_observed_all.append([])
    for k in range(4):
        normalized_observed_all[n].append(normalize(observed_all[n][k]))

normalized_observed_all = np.array(normalized_observed_all)

data = {
  "N": 56,
  "K": 4,
  "J": 4,
  "obs_e_prob": normalized_observed_all,
  "prior": np.array(pi_list[items])
}

for k, model_tag in enumerate(model_tags):
    sm = models[model_tag]
    mle_op = sm.optimize(data=data, seed=1)

    print('log likelihood evaluated at the MLE of the parameters for {} is:\n'.format(model2name[model_tag]), 
          mle_op.optimized_params_dict['log_lh'], get_log_lh(data['obs_e_prob'], mle_op.stan_variables()['inferred_e_prob'], mle_op.optimized_params_dict['sigma']))

    mle_op_all['combined'][model_tag] = mle_op.optimized_params_dict
    print()



# Print out log likelihood at the MLE estimation for each model
for exp_name in exp_names+['combined']:
    print_header('Estimated on data from {}'.format(exp_name))
    for k, model_tag in enumerate(model_tags):
        print('log likelihood evaluated at the MLE of the {} model: {}'.format(
            model_tag, mle_op_all[exp_name][model_tag]['log_lh']))
    print()


# Run likelihood ratio test between context-insensitive and full models
print_header('Likelihood ratio test results')
for exp_name in exp_names+['combined']:
    LR = likelihood_ratio(mle_op_all[exp_name]['context-general_lh']['log_lh'], mle_op_all[exp_name]['full']['log_lh'])
    p = chi2.sf(LR, 12-3) # L2 has 12-3 DoF more than L1
    print('Likelihood ratio test between context-general and full models on {} data'.format(exp_name))
    print('p:', p)
    print()
