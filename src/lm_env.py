import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from typing import List
import utils

from fast_detect_gpt.run_detector import FastDetectGPT

# args: max_sample_tokens
# args: model_name

class LMEnv:
    def __init__(self, args, initial_text):
        self.max_sample_tokens = args.max_sample_tokens
        self.model, self.tok = utils.get_model_and_tokenizer(args.model_name)
        assert isinstance(self.model, transformers.GPT2LMHeadModel)
        self.stop_tokens = utils.stop_tokens(self.tok)
        self._seed = None
        self.vocab_size = len(self.tok)
        # Current inputs and logits
        self.input_ids = self.tok(initial_text, return_tensors="pt")["input_ids"]
        self.cur_logits = None

    def get_text(self):
        return self.tok.decode(torch.squeeze(self.input_ids, dim=0))

    def sample_done(self):
        #TODO: give more stop tokens
        return self.input_ids[-1] in self.stop_tokens or self.cur_input.shape[1] >= self.max_sample_tokens

    def reset(self, new_input_ids):
        self.input_ids = new_input_ids
        self.cur_logits = None
        # TODO:Clear model cache

    def do_sample(
        self,
        max_tokens: int,
        sampling_mode: str = "likelihood"
    ) -> List[int]:
        sampled_tokens = []
        cum_logits = []
        n = 0
        cur_input = self.input_ids
        past_kvs = None
        with torch.inference_mode():
            while n < max_tokens:
                outputs = self.model(cur_input, past_key_values=past_kvs, use_cache=True)
                local_logits = outputs.logits[:, -1, :]
                cum_logits.append(local_logits)
                if sampling_mode == "argmax":
                    sampled_token = torch.argmax(local_logits, dim=-1)
                elif sampling_mode == "likelihood":
                    sampled_token = torch.multinomial(F.softmax(local_logits, dim=-1), num_samples=1)
                if sampled_token[0].item() in self.stop_tokens:
                    break
                sampled_tokens.append(sampled_token[0])
                cur_input = sampled_token.unsqueeze(0)
                past_kvs = outputs.past_key_values
                n += 1
        # (Batch Size=1, Input + Output Seq Len)
        new_tokens = torch.cat((self.input_ids, torch.tensor(sampled_tokens).unsqueeze(dim=0)), dim=1) 
        # (Seq Len, Batch Size=1, Vocab Size)
        new_logits = torch.stack(cum_logits, dim=0) 
        return new_tokens, new_logits
    
    def step(self, perturb=False, perturb_token=-1.):
        if perturb:
            self.input_ids = torch.cat(self.input_ids, torch.tensor(perturb_token).unsqueeze(dim=0).unsqueeze(dim=0), dim=1)
            logits = torch.zeros(self.vocab_size).float32()
            logits[perturb_token] = 1.
            self.cur_logits = logits.numpy()
        else:
            new_tokens, new_logits = self.do_sample(max_tokens=1)
            self.cur_logits = new_logits[-1, 0, :].numpy()
            self.input_ids = new_tokens


class FGPTEnv(gym.Env):

    def __init__(self, args, initial_text):
        self._seed = None
        # Environment
        self._env = LMEnv(args, initial_text)
        # Detector
        self._detector = FastDetectGPT(args)
        # This will be the logits 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.vocab_size,), dtype=np.float32)
        # Action Space, whether perturb or not


    def step(self, action):
        """
        :param action: One token selected from the token logits
        """
        if action["perturb"]:
            self._env.step(perturb=True, perturb_token=action["perturb_token"])
        else:
            self._env.step(perturb=False)
        
        observation = self._get_obs()
        rewards = -self._detector.infer(self._env.get_text())
        done = self._env.sample_done()
        info = None

    	# Within the same sentence, process the next word
        return observation, rewards, done, info

    def reset(self):
    	# Start a new sentence
        # self.env.reset()
        return self._get_obs()

    def close(self):
    	# Remove the entire FGPT_ENV from GPU
        # print("Closing environment")
        # self.env.close()
        pass

    def seed(self, seed=None):
        if seed:
            self._seed = np.random.seed(seed)

    # Private methods
    def _get_obs(self):
        return self._env.cur_logits.astype(np.float32)
