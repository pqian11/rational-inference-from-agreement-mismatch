import numpy as np
from cmdstanpy import CmdStanModel
import arviz
import json
import os.path
import time


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores, dtype='float')/total)


def get_error_freq_list(data_dict):
    conditions = ['SS_', 'SP_', 'PS_', 'PP_']
    e_types = ['subject','verb', 'subject+local', 'verb+local']
    rs = np.zeros((4, 4), dtype='int')
    for k, cond in enumerate(conditions):
        for j, e_type in enumerate(e_types):
            rs[k, j] = data_dict[cond][e_type]
    return rs.tolist()


def normalize_list(arr):
    rs = []
    for d in arr:
        rs.append(normalize(d))
    return rs


# Summarize Ryskin et al Exp1 data
e_dict_ryskin_exp1a = json.load(open('data/ryskin_et_al_exp1a_error_count.json'))

# Summarize Kandel et al Exp1 data
e_dict_kandel_exp1= json.load(open('data/kandel_et_al_exp1_error_count.json'))

# Summarize Kandel et al Exp2 data
e_dict_kandel_exp2= json.load(open('data/kandel_et_al_exp2_error_count.json'))

model_path = 'stan_model/meta_analysis.stan'
sm = CmdStanModel(stan_file=model_path)

data = {
  "N": 3, # three studies
  "K": 4,
  "J": 4,
  "error_freq": [get_error_freq_list(e_dict_ryskin_exp1a), get_error_freq_list(e_dict_kandel_exp1), get_error_freq_list(e_dict_kandel_exp2)]
}

# print(data["error_freq"])

fit = sm.sample(data=data, chains=4, iter_warmup=10000, iter_sampling=10000, refresh=1000, max_treedepth=15, adapt_delta=0.99, show_progress=True, show_console=True, seed=1122)
# fit.save_csvfiles(dir="stan_model/mcmc_output")

start_time = time.time()
print('Run diagonistics')
print(fit.diagnose())
print('Diagnoistics takes', (time.time() - start_time), 's')

summary_df = fit.summary()
print(summary_df.head(50))
# print(summary_df.tail(20))

start_time = time.time()
model_fit = arviz.from_cmdstanpy(fit)
print((time.time() - start_time), 's')

savepath = 'stan_model/model_fit/meta_analysis_model_fit.nc'
arviz.to_netcdf(model_fit, savepath)

