from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    try:
        obj = cls.from_pretrained(local_path, **kwargs)
    except Exception as ex:
        print(ex)
        obj = cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)
        obj.save_pretrained(local_path)
    return obj

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json




def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    # assert logits_ref.shape[0] == 1
    # assert logits_score.shape[0] == 1
    # assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    # print(logits_ref.shape, logits_score.shape, labels.shape)
    # (1, 95, 50257), (1, 95, 50257), (1, 95)
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    # print(log_likelihood.shape, mean_ref.shape, var_ref.shape)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    # print(discrepancy.shape)
    return discrepancy


import random

import numpy as np
import torch
import os
import glob
import argparse
import json
import transformers
import datasets



# reference_model_name = "gpt-j-6B"
# scoring_model_name = "gpt-neo-2.7B"

reference_model_name = "gpt2"
scoring_model_name = "gpt2"


dataset = "xsum"
ref_path = "./local_infer_ref"
device = "cuda"
cache_dir = "../cache"

class ProbEstimator:
    def __init__(self):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')
        self.real_crits_tensor = torch.tensor(self.real_crits).to(device)
        self.fake_crits_tensor = torch.tensor(self.fake_crits).to(device)


    def crit_to_prob(self, crit):
        real_crits_tensor = self.real_crits_tensor.unsqueeze(dim=1)
        fake_crits_tensor = self.fake_crits_tensor.unsqueeze(dim=1)

        real_diffs = torch.abs(torch.cat([real_crits_tensor, fake_crits_tensor]) - crit.unsqueeze(dim=0))
        # print(crit.shape, real_diffs.shape)

        # Calculate offset
        offset, _ = torch.sort(real_diffs, dim=0)
        offset = offset[100, :]

        # Count occurrences using PyTorch operations
        lower_bound = (crit - offset).unsqueeze(dim=0)
        upper_bound = (crit + offset).unsqueeze(dim=0)
        cnt_real = torch.sum((real_crits_tensor > lower_bound) & (real_crits_tensor < upper_bound), dim=0)
        cnt_fake = torch.sum((fake_crits_tensor > lower_bound) & (fake_crits_tensor < upper_bound), dim=0)
        # Convert to float for division
        cnt_real = cnt_real.float()
        cnt_fake = cnt_fake.float()

        # Calculate and return the probability
        return cnt_fake / (cnt_real + cnt_fake)

        # offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        # cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        # cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        # return cnt_fake / (cnt_real + cnt_fake)



class FastDetectGPT:
    def __init__(self):
        self.device = device
        # load model
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        self.scoring_model = load_model(scoring_model_name, device, cache_dir)
        self.scoring_model.eval()
        self.reference_model_name = reference_model_name
        self.scoring_model_name = scoring_model_name
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.reference_model_name, dataset, cache_dir)
            self.reference_model = load_model(self.reference_model_name, device, cache_dir)
            self.reference_model.eval()
        # evaluate criterion
        self.criterion_name = "sampling_discrepancy_analytic"
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.prob_estimator = ProbEstimator()
        # input text
        print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
        print('')

    def infer(self, text):
        # evaluate text     # (1, 112)
        tokenized = self.scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        prob = self.prob_estimator.crit_to_prob(crit)
        # print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
        return prob


detector = FastDetectGPT()



from typing import List, Set

def model2hfname(model: str) -> str:
    return {
        "bert-tiny": "prajjwal1/bert-tiny",
        "bert-med": "prajjwal1/bert-medium",
        "small": "gpt2",
        "med": "gpt2-medium",
        "large": "gpt2-large",
        "full": "gpt2-xl",
        "gpt2-sm": "gpt2",
        "gpt2-med": "gpt2-medium",
        "gpt2-lg": "gpt2-large",
        "gpt2": "gpt2-xl",
        "neo": "EleutherAI/gpt-neo-2.7B",
    }[model]

def get_model_and_tokenizer(model: str, Cls = transformers.AutoModelForCausalLM, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({"pad_token": "[PAD]"})
            tok.pad_token = "[PAD]"
    return m, tok


def stop_tokens(tokenizer, stop_strings: Set[str] = set(".")) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Stop tokens:", tokens)
    return tokens

def ignore_tokens(tokenizer, stop_strings: Set[str] = set("\n")) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Ignore tokens:", tokens)
    return tokens

def ignore_tokens_replace(tokenizer, stop_strings: Set[str] = set(" ")) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Ignore tokens replaced by:", tokens)
    return tokens[0]

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

