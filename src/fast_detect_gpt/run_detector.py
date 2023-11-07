
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

class FastDetectGPT:
    def __init__(self, args):
        self.device = args.device
        # load model
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        self.reference_model_name = args.reference_model_name
        self.scoring_model_name = args.scoring_model_name
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.reference_model_name, args.dataset, args.cache_dir)
            self.reference_model = load_model(self.reference_model_name, args.device, args.cache_dir)
            self.reference_model.eval()
        # evaluate criterion
        self.criterion_name = "sampling_discrepancy_analytic"
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.prob_estimator = ProbEstimator(args)
        # input text
        print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
        print('')

    def infer(self, text):
        # evaluate text
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
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
        return prob
