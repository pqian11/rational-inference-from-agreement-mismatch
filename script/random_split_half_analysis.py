import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.rcParams['font.family'] = 'arial'
import pandas as pd
import scipy.stats
from sklearn.metrics import r2_score


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


def get_observed_freq(df):
    # Get edit type counts for each condition of every item
    observed_all = np.zeros((len(items), len(conditions), len(e_types)))
    for row_idx, row in df.iterrows():
        item = row['item']
        trial_condition = row['trial_condition']
        e_type = row['e_type']
        observed_all[item2idx[item], condition2idx[trial_condition], e_type2idx[e_type]] += 1
    return observed_all


def plot_hist(rs, xlabel, title=None, n_bin=10, savefig=None):
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.hist(rs, bins=n_bin, color='grey', density=True)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    
    if title:
        plt.title(title)
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()   


exp_names = ['exp1', 'exp2']
pretty_names = ['Study I', 'Study II']

e_types = ['subj', 'verb', 'subj+local', 'verb+local']
e_type2idx = dict([(e_type, i) for i, e_type in enumerate(e_types)])
items = [i for i in range(57) if i != 55]
item2idx = dict([(item, i) for i, item in enumerate(items)])
conditions = ['SSP', 'SPP', 'PSS', 'PPS']
condition2idx = dict([(condition, i) for i, condition in enumerate(conditions)])

obs_list = []

df_all_list = []

for exp_idx, exp_name in enumerate(exp_names):
    # Load data frame
    df_all = pd.read_csv('data/{}_df_target_mismatch_resolving_trials.csv'.format(exp_name), index_col=0)
    df_all_list.append(df_all)


df_combined = pd.concat(df_all_list, ignore_index=True, sort=False)

# Create a mask list
cut1 = int(len(df_combined)/2)
cut2 = len(df_combined) - cut1
print(cut1, cut2)
msk = np.array([True for _ in range(cut1)] + [False for _ in range(cut2)])

np.random.seed(10)

mse_all = []
rho_all = []
r2_all = []

i = 0

while i < 3000:
    np.random.shuffle(msk)
    obs_list = [get_observed_freq(df_combined[msk]), get_observed_freq(df_combined[~msk])]

    xs = []
    ys = []

    # Check that each item-condition combination has at least one data point in the random split
    flag = False
    for n, item in enumerate(items):
        for k, condition in enumerate(conditions):
            
            if np.sum(obs_list[0][n][k]) == 0 or np.sum(obs_list[1][n][k]) == 0:
                flag = True
                continue
            
            xs += normalize(obs_list[0][n][k])
            ys += normalize(obs_list[1][n][k])

    if flag:
        continue
            
    mse = np.square(np.array(xs) - np.array(ys)).mean()
    rho, p_value = scipy.stats.spearmanr(xs, ys)
    r2 = r2_score(ys, xs)
    
    mse_all.append(mse)
    rho_all.append(rho)
    r2_all.append(r2)
    
    i += 1
    

plot_hist(mse_all, 'MSE', 'Distribution of MSE', n_bin=20, savefig='fig/dist_of_mse_from_random_half_split_of_combined_data.pdf')  
plot_hist(rho_all, r'$\rho$', r'Distribution of $\rho$', n_bin=20, savefig='fig/dist_of_spearmanr_from_random_half_split_of_combined_data.pdf')  
plot_hist(r2_all, r'$R^2$', n_bin=20)  
