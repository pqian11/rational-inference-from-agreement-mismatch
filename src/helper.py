import nltk
import numpy as np

def formatted_response(stimulus, answers):
    formatted_str = ''
    tokens = stimulus.split('%')
    for i in range(len(tokens)):
        if i % 2 == 0:
            if tokens[i] == '':
                continue
            if i == 0:
                formatted_str += tokens[i].strip()+' '
            elif i == len(tokens)-1 and tokens[i].strip() == '.':
                formatted_str += tokens[i].strip()
            else:
                formatted_str += ' '+tokens[i].strip()+' '
        else:
            j = i // 2
            formatted_str += answers[j]
    return formatted_str


def normalize(scores):
    total = np.sum(scores)
    return np.array(scores)/total


def map_to_feature(sent, stimulus):
    features = ['S', 'P']
    pred_dict = {'S':set(['is', 'was', 'has']), 'P':set(['are', 'were', 'have'])}
    verb2feature = {'is':'S', 'was':'S', 'has':'S', 'are':'P', 'were':'P', 'have':'P'}
    words = nltk.word_tokenize(sent.lower())
    word_set = set(words)
    N1s, N2s, preds = stimulus['choices']
    structure = ['O', 'O', 'O']
    for k, options in enumerate(stimulus['choices']):
        for i in range(len(options)):
            if k < 2:
                token = options[i]
            else:
                # Only use the first token of the predicate
                token = options[i].split()[0]
                    
            if token in word_set:
                structure[k] = features[i]
                break
                
    if structure[-1] == 'O':
        tagged = nltk.pos_tag(words)
        for w, tag in tagged:
            if tag == 'VBG':
                continue
            elif tag.startswith('VB'):
                if w in verb2feature:
                    structure[-1] = verb2feature[w]
                else:
                    if tag == 'VBD':
                        structure[-1] = structure[0]
                    elif tag == 'VBP' or tag == 'VB':
                        structure[-1] = 'P'
                    elif tag == 'VBZ':
                        structure[-1] = 'S'
                    else:
                        pass
                break
        
    return ''.join(structure)


def identify_error_type(obs, hypothesis):
    """
    Given an observation (one of ['SSP', 'SPP', 'PSS', 'PPS'])
    and the intended/post-edit form, identify the error type.
    """
    if (obs[0] != hypothesis[0]) and (obs[1] == hypothesis[1]) and (obs[2] == hypothesis[2]):
        possible_e = 'subj'
    elif (obs[0] == hypothesis[0]) and (obs[1] == hypothesis[1]) and (obs[2] != hypothesis[2]):
        possible_e = 'verb'
    elif (obs[0] != hypothesis[0]) and (obs[1] != hypothesis[1]) and (obs[2] == hypothesis[2]):
        possible_e = 'subj+local'
    elif (obs[0] == hypothesis[0]) and (obs[1] != hypothesis[1]) and (obs[2] != hypothesis[2]):
        possible_e = 'verb+local'
    else:
        possible_e = 'other'
    return possible_e
