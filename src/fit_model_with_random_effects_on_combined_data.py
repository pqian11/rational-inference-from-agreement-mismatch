import numpy as np
from cmdstanpy import CmdStanModel
import arviz
import json
import pickle
import pandas as pd
import os.path
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model", help="Name of the model to be fitted.", type=str)
parser.add_argument('--do_compile', help="Recompile the STAN model.", action='store_true')
args = parser.parse_args()


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


DO_COMPILE = 'force' if args.do_compile else True # If do_compile is true, then use 'force' option for the compile flag. Otherwise, use True.

# Load estimated prior
pi_list = json.load(open('data/estimated_prior_from_norming_task.json'))
pi_list = np.array(pi_list)


model2name = {"prior-only":"bayesian_model_prior-only_temperature_with_random_effect",
                "lh-only":"bayesian_model_lh-only_with_random_effect",
                "context-general_lh":"bayesian_model_context-general_lh_temperature_with_random_effect",
                "full":"bayesian_model_full_temperature_with_random_effect"}
model_name = model2name[args.model]
model_path = 'stan_model/{}.stan'.format(model_name)

sm = CmdStanModel(stan_file=model_path)


e_type_list = []
item_list = []
subject_list = []
condition_list = []

exp_names = ['exp1', 'exp2']

for exp_name in exp_names:
    e_types = ['subj', 'verb', 'subj+local', 'verb+local']
    e_type2index = dict([(e_type, i+1) for i, e_type in enumerate(e_types)])
    items = [k for k in range(57) if k != 55]
    item2index = dict([(item, i+1) for i, item in enumerate(items)])
    conditions = ['SSP', 'SPP', 'PSS', 'PPS']
    condition2index = dict([(condition, i+1) for i, condition in enumerate(conditions)])

    fpath = 'data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name)
    df = pd.read_csv(fpath)

    print('Load trial data from {}'.format(fpath))

    e_type_list += [e_type2index[e_type] for e_type in df['e_type'].to_list()]
    item_list += [item2index[item] for item in df['item'].to_list()]
    if exp_name == 'exp1':
        subject_list += [subject+1 for subject in df['subject'].to_list()]
    else:
        subject_list += [98+subject+1 for subject in df['subject'].to_list()]
    condition_list += [condition2index[condition] for condition in df['trial_condition'].to_list()]

print(len(e_type_list))


data = {
  "N": 56,
  "K": 4,
  "J": 4,
  "M": int(np.max(subject_list)),
  "L": len(e_type_list),
  "e_type_list": np.array(e_type_list, dtype='int').tolist(),
  "subject_list": np.array(subject_list, dtype='int').tolist(),
  "item_list": np.array(item_list, dtype='int').tolist(),
  "condition_list": np.array(condition_list, dtype='int').tolist(),
  "prior": np.array(pi_list[items]).tolist(),
  "lambda": 0.1
}

fit = sm.sample(data=data, chains=4, iter_warmup=2000, iter_sampling=8000, show_progress=True, show_console=True, seed=101)

fit.save_csvfiles(dir="stan_model/mcmc_output")

start_time = time.time()
model_fit = arviz.from_cmdstanpy(fit)
print((time.time() - start_time), 's')

savepath = 'stan_model/model_fit/exp_combined_{model_name}_model_fit.nc'.format(model_name=model_name)
arviz.to_netcdf(model_fit, savepath)


