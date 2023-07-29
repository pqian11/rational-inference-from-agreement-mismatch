import numpy as np
import pandas as pd
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer


def get_score(model, tokenizer, sent, device='cuda', tokenized=True):
    if tokenized:
        line = TreebankWordDetokenizer().detokenize(sent)
    else:
        line = sent

    input_ids = tokenizer.encode(line, return_tensors='pt').to(device)
    outputs = model(input_ids, labels=input_ids)

    score = outputs[0].item()*(input_ids.size()[1] - 1)
    return score


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt_model_version = 'gpt2'

# Load pre-trained GPT-2 tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_version, cache_dir='./pretrained')

# Load pre-trained GPT-2 model
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_version, cache_dir='./pretrained')
gpt_model.eval()

if device == 'cuda':
    gpt_model = gpt_model.cuda(0)

# Prepare prior norming stimuli
infpath = 'data/prior_norming_stimuli.txt'

df = pd.read_csv(infpath)

stimuli = []

item_set = set([k for k in range(57) if k != 55])

for i, row in df.iterrows():
    if i not in item_set:
        continue

    N1s = row['N1'].split('/')
    N2s = row['N2'].split('/')
    preposition = row['preposition']
    context = row['context'].format(preposition=row['preposition'], predicate=row['predicate'])

    group = N1s[0] + '_' + N2s[0]
    choices = [N1s, N2s]

    sent_set = []
    for m in range(len(N1s)):
        for n in range(len(N2s)):
            context = row['context'].format(preposition=row['preposition'], predicate=row['predicate']).replace('%%', '{}')
            sent = context.format(N1s[m], N2s[n])
            sent_set.append(sent)
    stimuli.append(sent_set)


# Evaluate log probs of sentences and normalize within each item
prior_probs_all = []

for i in range(len(stimuli)):
    sent_set = stimuli[i]
    logprobs = []
    for sent in sent_set:
        score = get_score(gpt_model, gpt_tokenizer, sent, tokenized=False)
        logprobs.append(-score)

    gpt_probs = exp_normalize(np.array(logprobs))
    prior_probs_all.append(list(gpt_probs))
    print(gpt_probs)

json.dump(prior_probs_all, open('data/gpt2_estimated_prior.json', 'w'), indent=4)
