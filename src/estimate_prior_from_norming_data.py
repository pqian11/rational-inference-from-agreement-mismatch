import json
import numpy as np
import pandas as pd


def normalize(scores):
    total = np.sum(scores)
    return np.array(scores)/total


norming_data = json.load(open('data/norming_task_participant_data.json'))


infpath = 'data/prior_norming_stimuli.txt'
df = pd.read_csv(infpath)

stimuli = []
group2idx = {}

for i, row in df.iterrows():
    N1s = row['N1'].split('/')
    N2s = row['N2'].split('/')
    # preposition = df['preposition']
    context = row['context'].format(preposition=row['preposition'], predicate=row['predicate'])

    group = N1s[0] + '_' + N2s[0]
    choices = [N1s, N2s]

    stimuli.append({'group':group, 'choices':choices, 'stimulus':context, 'choices':choices})
    group2idx[group] = i


priors = {}
priors_normalized = {}
conditions = ['SS', 'SP', 'PS', 'PP']
for stimulus_idx, stimulus in enumerate(stimuli):
    group = stimulus['group']
    priors[group] = {}
    priors_normalized[group] = {}
    for condition in conditions:
        priors[group][condition] = []
        priors_normalized[group][condition] = []
    
exclude_subject_ids = []
    
for i, subject_data in enumerate(norming_data):
    for trial in subject_data[1:]:      
        if trial['trial_type'] == 'cloze-with-slider':
            context = trial['context']
            group = trial['group']
            
            scores = []

            for j, condition in enumerate(conditions):
                response = trial['answers'][condition]
                if response.startswith('<'):
                    response = BeautifulSoup(response).get_text()
                priors[group][condition].append(float(response))
                scores.append(float(response))
                
            normalized_scores = normalize(scores)
            for j, condition in enumerate(conditions):
                priors_normalized[group][condition].append(normalized_scores[j])
                

            if group.startswith('rice'):
                if scores[2] > 50 or scores[3] > 50:
                    exclude_subject_ids.append(i)
            
print("Index list of excluded subjects:", exclude_subject_ids)

obs_all = []
for stimulus_idx, stimulus in enumerate(stimuli):
    group = stimulus['group']
    stimulus_idx = group2idx[group]
    stimulus = stimuli[stimulus_idx]
    
    obs = []
    n_subject = len(priors_normalized[group][conditions[0]])
    for i in range(n_subject):
        if i in exclude_subject_ids:
            continue
        subject_rating = [priors_normalized[group][condition][i] for condition in conditions]
        obs.append(subject_rating)

    obs_all.append(obs)

obs_all = np.array(obs_all)

print('Estimating prior based on rating from {} subjects.'.format(len(obs_all[0])))

for stimulus_idx, stimulus in enumerate(stimuli):
    print(stimulus_idx, np.mean(obs_all[stimulus_idx], axis=0))

estimated_prior = [list(np.mean(obs, axis=0)) for obs in obs_all]

print('Export estimated prior from norming task to json file.')
json.dump(estimated_prior, open('data/estimated_prior_from_norming_task.json', 'w'), indent=4)
