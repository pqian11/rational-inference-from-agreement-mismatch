import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "arial"
import scipy.stats


def normalize(scores):
    total = np.sum(scores)
    return list(np.array(scores)/total)


conditions = ['SS_', 'SP_', 'PS_', 'PP_']
items = [k for k in range(57) if k != 55]

# Load estimated prior from human norming experiment
pi_list = json.load(open('data/estimated_prior_from_norming_task.json'))
pi_list = [pi_list[item] for item in items]

# Load prior estimated from GPT-2
gpt_priors = json.load(open('data/gpt2_estimated_prior.json'))

# Plot model and human comparison
plt.figure(figsize=(4,4))
ax = plt.gca()
xs = [x for i in range(len(items)) for x in pi_list[i]]
ys = [y for i in range(len(items)) for y in gpt_priors[i]]
plt.plot(xs, ys, 'o', color='grey', mfc='none', alpha=0.7)
plt.xlabel('Human')
plt.ylabel('GPT-2')
plt.xlim(xmin=0)
plt.ylim(0, 1.02)

print('Pearson r:', scipy.stats.pearsonr(xs, ys))
print('Spearman rho:', scipy.stats.spearmanr(xs, ys))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.text(0.02, 0.95, r'$\rho=${:.3f}***'.format(scipy.stats.spearmanr(xs, ys)[0]))


plt.savefig('fig/human_gpt2_prior_comparison.pdf', bbox_inches='tight')
plt.show()