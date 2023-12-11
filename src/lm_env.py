import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from typing import List
from utils import *
from fast_detect_gpt.run_detector import FastDetectGPT


import gymnasium as gym
import tqdm

class LMEnv(gym.Env):
    def __init__(self, args):

        self.args=args

        # Batch Size
        self.batch_size = self.args.batch_size
  
        # Dataset
        self.data_items = self.args.data_items.split(',')
        self.data_items = torch.tensor(list(map(lambda x:int(x), self.data_items))).to(self.args.env_device)
        assert(len(self.data_items) == self.batch_size)
        self._seed = self.args.random_seed
        self.dataset = self.args.dataset
        self.n_train = self.args.n_train
        self._load_datasets()

        ## LLM
        self.max_sample_tokens = self.args.max_sample_tokens
        self.model, self.tok = get_model_and_tokenizer(self.args.model_name)
        assert isinstance(self.model, transformers.GPT2LMHeadModel)
        self.model.to(self.args.env_device)
        self.stop_tokens = stop_tokens(self.tok)
        self.ignore_tokens = ignore_tokens(self.tok)
        self.ignore_tokens_replace = ignore_tokens_replace(self.tok)
        self.vocab_size = len(self.tok)
        # Current inputs and logits

        self.topK_logistics = self.args.topK_logistics
        

        self.sampling_mode = self.args.sampling_mode  # "likelihood" or "argmax"
        self.num_perturb = None
        self.past_obs = None

        self.input_ids = None
        self.attention_mask = None
        self.output_mask = None
        self.input_mask = None
        self.past_kvs = None
        self.last_logits = None
        self.last_logits_unperturbed = None
        self.input_ids_unperturbed = None
        self.past_kvs_unperturbed = None
        self.sample_done = None

        ## RL: Basic Action Space and Obs Space
        # Whether perturb or not.
        # If not perturb: sample by multinomial
        # If perturb: sample by equal probability
        self.obs_dim = self.args.obs_dim
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, self.topK_logistics), dtype=np.float32)

        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(f"CS330_FastGPT_{model_name}_{env_device}/{algorithm}/OLD_Action_MLogits")
        self.zero_tensor = torch.zeros(self.batch_size).to(self.args.env_device)
        self.one_tensor = torch.ones(self.batch_size).to(self.args.env_device)

        self.reset(random=self.args.random)

    def _load_datasets(self):
        print("Dataset:", self.dataset)
        if self.dataset == "xsum":
            d = datasets.load_dataset(self.dataset, split="train").shuffle(seed=self._seed)
            filter_fn = lambda rows: [
                len(a.split(" ")) < 100 for a in rows["document"]
            ]
            d = d.filter(filter_fn, batched=True, batch_size=None)
            d = d["document"][:self.n_train]
            self.data = d
        else:
            raise NotImplementedError

    def _get_new_input(self, items):
        ret = []
        for item in items:
            ret.append(self.data[item].replace('\n', ' '))
        return ret

    def _feedforward(self, cur_input, attention_mask, past_kvs=None):
        # TODO: Speed up feedforward by utilizing past_kvs
        """
        :param cur_input: When past_kvs = None, tensor shape [batch_size, seq_len]. When past_kvs is not None, tensor shape [batch_size, 1]
        :param past_kvs: a cache to speed up model inference
        :return local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :return new_past_kvs: the new model state cache
        """
        with torch.inference_mode():
            outputs = self.model(cur_input,
                                #  past_key_values=past_kvs,
                                 attention_mask=attention_mask,
                                 # use_cache=True
                                 use_cache=False)
            all_logits = outputs.logits
            B, S, V = all_logits.shape
            returned_logits = torch.ones(B, self.obs_dim, V).float().to(self.args.env_device)
            if S < self.obs_dim:
                returned_logits[:, self.obs_dim - S:, :] = all_logits
            else:
                returned_logits = all_logits[:, S - self.obs_dim:, :]
            # new_past_kvs = outputs.past_key_values
            new_past_kvs = None
            return returned_logits, new_past_kvs

    def _cat_new_word(self, sampled_token, input_ids):
        token_len = sampled_token.shape[0]
        return torch.cat((input_ids, sampled_token.clone().detach().long().view(-1, 1)), dim=1)

    def _sample_tokens(self, local_logits, input_ids, attention_mask):
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits
         at the last point
        :param input_ids: tensor shape [batch_size, seq_len] input ids at latest
         point
        :param attention_mask: tensor shape [batch_size, seq_len] attention
         mask at latest point
        :return new_token: tensor shape [batch_size, 1]
        works together with past_kvs returned from get_logits() to feed in the
         next round of get_logits().
        :return new_input_ids: when past_kvs = None, this would return the
         complete input concat with output up to this point
        :return new_attention_mask: attention mask extended
        """
        if self.sampling_mode == "argmax":
            sampled_token = torch.argmax(local_logits, dim=-1)
        elif self.sampling_mode == "likelihood":
            sampled_token = torch.multinomial(F.softmax(local_logits, dim=-1), num_samples=1).squeeze(dim=1)
        else:
            raise NotImplementedError

        # Replace tokens such as new line with spaces

        mask = torch.any(torch.eq(sampled_token, torch.tensor(self.ignore_tokens).to(self.args.env_device)), dim=-1)
        sampled_token[mask] = self.ignore_tokens_replace

        new_token = sampled_token.view(-1, 1)
        new_input_ids = self._cat_new_word(new_token, input_ids)
        new_attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        return new_token, new_input_ids, new_attention_mask

    def _perturb_tokens(self, local_logits, perturb_mode="chosen", perturb_ranking=-1):
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :param perturb_ranking: perturb selection of the last word
        :return new_token: the selected token to generate
        :return new_input_ids: the new input ids after the perturbation
        """
        # Get the top k predictions （1-10）
        if perturb_mode == "chosen":
            _, topk_indices = torch.topk(local_logits, perturb_ranking, dim=1)
            # Select the last item
            new_token = topk_indices[:, -1]
            new_input_ids = self._cat_new_word(new_token, self.input_ids)
            return new_token, new_input_ids
        else:
            _, topk_indices = torch.topk(local_logits, 10, dim=1)
            # Select random item
            new_token = topk_indices[:, random.randint(0, 9)]
            new_input_ids = self._cat_new_word(new_token, self.input_ids)
            return new_token, new_input_ids

    def _obs_wrapper(self, all_logits):
        """
        :param all_logits: tensor shape [batch_size, seq_len, vocab_size]
        :return topk_values: numpy array shape [batch_size, seq_len, topk]
        :return topk_indices: numpy array shape [batch_size, seq_len, topk]
        """
        topk_values, topk_indices = torch.topk(all_logits, self.topK_logistics, dim=-1)
        # Normalize the topk_values
        topk_values = F.softmax(topk_values, dim=-1)

        return topk_values.detach().cpu().numpy(), topk_indices.detach().cpu().numpy()

    def _sample_done(self):
        """
        ERROR PRONE: Do not use unless validated.
        """
        input_ids = self.input_ids[:,-1].unsqueeze(dim=-1)
        stop_tokens = torch.tensor(self.stop_tokens).view(1, -1).to(self.args.env_device)

        a = torch.any(torch.eq(input_ids, stop_tokens))

        b = self.input_ids.shape[1] >= self.max_sample_tokens
        b = torch.tensor(b).repeat((self.batch_size)).to(self.args.env_device)
        # print("Done:", a, input_ids[0][0], stop_tokens, b)
        return a | b

    def _masked(self, a, b, mask):
        """
        ERROR PRONE: Do not use unless validated.
        """
        if a is None:
            return b
        else:
            if a.shape[-1] > b.shape[-1]:
                b_zeros = torch.zeros_like(a)
                b_zeros[...,:b.shape[-1]] = b
                b = b_zeros
            else:
                a_zeros = torch.zeros_like(b)
                a_zeros[...,:a.shape[-1]] = a
                a = a_zeros
            a[mask] = b[mask]
            return a

    def _reset(self, random=True, mask=None):
        """
        :param random: whether to sample data randomly
         if random == False, choose self.data_items in the dataset
        :param mask: only reset these rows in the batch
        ERROR PRONE: Do not use unless debugged
        """
        print("Reset begins...")
        if mask is None:
            mask = self.one_tensor.bool().detach()
        if not torch.any(mask):
            return self.past_obs, None
        if random:
            data_items = torch.randint(low=0, high=self.n_train, size=(self.batch_size,)).to(env_device)
            self.data_items = self._masked(self.data_items, data_items, mask)
        print("Data Items:", self.data_items, "Mask:", mask)

        ## Get a new generate starting point
        initial_texts = self._get_new_input(self.data_items)

        be = self.tok(initial_texts,
                              return_tensors="pt",
                              padding=True, return_attention_mask=True)

        self.input_ids = self._masked(self.input_ids,
                     be["input_ids"]
                     .to(self.args.env_device),
                     mask)
        self.attention_mask = self._masked(self.attention_mask,
                     be["attention_mask"]
                     .to(self.args.env_device),
                     mask)
        self.num_perturb = self._masked(self.num_perturb, self.zero_tensor.clone().detach(), mask)
        while random and self.input_ids.shape[-1] ==0:
            self.data_items = self._masked(self.data_items,
                         np.random.randint(self.n_train, size=self.batch_size),
                         mask)
            initial_texts = self._get_new_input(self.data_items)
            be = self.tok(initial_texts,
                          return_tensors="pt",
                          padding=True,
                          return_attention_mask=True)
            self.input_ids = self._masked(self.input_ids,
                        be["input_ids"]
                        .to(self.args.env_device),
                        mask)
            self.attention_mask = self._masked(self.attention_mask,
                        be["attention_mask"]
                        .to(self.args.env_device),
                        mask)
        ## First 1 step
        all_logits, new_past_kvs = self._feedforward(self.input_ids, self.attention_mask)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs
        self.sample_done = self.zero_tensor.clone().detach().bool()

        _, new_input_ids, new_attention_mask = self._sample_tokens(local_logits, self.input_ids, self.attention_mask)
        self.input_ids = new_input_ids
        self.attention_mask = new_attention_mask

        self.max_sample_tokens = max_sample_tokens
        if self.input_ids.shape[-1] + 20 > self.max_sample_tokens:
          self.max_sample_tokens = self.input_ids.shape[-1] + 20

        self.last_logits_unperturbed = self.last_logits
        self.past_kvs_unperturbed = self.past_kvs
        self.input_ids_unperturbed = self.input_ids
        self.attention_mask_unperturbed = self.attention_mask
        obs, _ = self._obs_wrapper(all_logits)

        ## NOTE: save the past obs
        self.past_obs = obs

        reset_info = {"TimeLimit.truncated": self.zero_tensor.clone().detach().bool(),
                      "DataItem": self.data_items,
                      "F_GPT_Score_drop": self.zero_tensor.clone().detach(),
                      "RL_num_perturb": self.zero_tensor.clone().detach(),
                      "last_reward": self.zero_tensor.clone().detach(),
                      }
        print("Reset ends!")
        # print(obs, reset_info)
        return obs, reset_info

    def _reset_all(self, random=True):
        """
        :param random: whether to sample data randomly
         if random == False, choose self.data_items in the dataset
        :param mask: only reset these rows in the batch
        """

        if random:
            data_items = torch.randint(low=0, high=self.n_train, size=(self.batch_size,)).to(env_device)
            self.data_items = data_items
        print("Reset All begins...", random, self.data_items)

        ## Get a new generate starting point
        initial_texts = self._get_new_input(self.data_items)

        batch_encoding = self.tok(initial_texts,
                      return_tensors="pt",
                      padding=True)

        self.input_ids = batch_encoding["input_ids"].to(self.args.env_device)
        self.attention_mask = batch_encoding["attention_mask"].to(self.args.env_device)
        self.num_perturb = self.zero_tensor.clone().detach()
        while random and self.input_ids.shape[-1] ==0:
            self.data_items = torch.randint(low=0, high=self.n_train, size=(self.batch_size,)).to(env_device)
            initial_texts = self._get_new_input(self.data_items)
            batch_encoding = self.tok(initial_texts,
                      return_tensors="pt",
                      padding=True)
            self.input_ids = batch_encoding["input_ids"].to(self.args.env_device)
            self.attention_mask = batch_encoding["attention_mask"].to(self.args.env_device)
        ## First 1 step
        all_logits, new_past_kvs = self._feedforward(self.input_ids, self.attention_mask)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        self.sample_done = self.zero_tensor.clone().detach().bool()

        _, new_input_ids, new_attention_mask = self._sample_tokens(local_logits, self.input_ids, self.attention_mask)
        self.input_ids = new_input_ids
        self.attention_mask = new_attention_mask
        self.output_mask = self.attention_mask
        self.input_mask = self.attention_mask

        self.last_logits_unperturbed = self.last_logits
        self.past_kvs_unperturbed = self.past_kvs
        self.input_ids_unperturbed = self.input_ids
        self.attention_mask_unperturbed = self.attention_mask

        self.max_sample_tokens = self.args.max_sample_tokens
        if self.input_ids.shape[-1] + 20 > self.max_sample_tokens:
          self.max_sample_tokens = self.input_ids.shape[-1] + 20

        print("Max tokens:", self.max_sample_tokens)

        obs, _ = self._obs_wrapper(all_logits)

        ## NOTE: save the past obs
        self.past_obs = obs

        reset_info = {"TimeLimit.truncated": self.zero_tensor.clone().detach().bool(),
                      "DataItem": self.data_items,
                      "F_GPT_Score_drop": self.zero_tensor.clone().detach(),
                      "RL_num_perturb": self.zero_tensor.clone().detach(),
                      "last_reward": self.zero_tensor.clone().detach(),
                      }
        print("Reset All ends!")
        # print(obs, reset_info)
        return obs, reset_info

    def reset(self, seed: int = None, random=True, mask = None):
        # print("Resetting environment=============")
        if mask == None:
            return self._reset_all(random=random)
        else:
            return self._reset(random=random, mask=mask)
        # return obs

    def get_texts(self, mask=None):
        """
        :return texts: str list [batch_size]
        """
        input_ids = self.input_ids.clone().detach()
        if mask is not None:
          input_ids[mask == 1] = self.tok.pad_token_id
        return self.tok.batch_decode(input_ids, skip_special_tokens=True)

    def get_texts_unperturbed(self, mask=None):
        """
        :return texts: str list [batch_size]
        """
        input_ids_unperturbed = self.input_ids_unperturbed.clone().detach()
        if mask is not None:
          input_ids_unperturbed[mask == 1] = self.tok.pad_token_id
        return self.tok.batch_decode(input_ids_unperturbed, skip_special_tokens=True)

    def _step_sample(self, perturb):
        """
        :param perturb: boolean tensor of shape [batch_size]
        :return obs: tensor of shape [batch_size, obs_dim, topk]
        :return done: bool tensor of shape [batch_size]
        """
        sampled_token, sampled_output, sampled_attention_mask = self._sample_tokens(self.last_logits, self.input_ids, self.attention_mask)

        _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_mode="chosen", perturb_ranking=self.args.perturb_ranking)

        self.input_ids = torch.where(perturb.unsqueeze(dim=-1), perturbed_output, sampled_output)
        self.attention_mask = sampled_attention_mask

        cur_input = self.input_ids
        self.past_kvs = None

        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.attention_mask, self.past_kvs)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        obs, token = self._obs_wrapper(all_logits)
        token = torch.tensor(token)[:, -1, :].unsqueeze(dim=-1).to(self.args.env_device)
        stop_tokens = torch.tensor(self.stop_tokens).view(1, 1, -1).to(self.args.env_device)

        done = torch.any(torch.eq(token, stop_tokens), dim=-1)
        done = torch.any(done, dim=-1)

        return obs, done

    def _step_sample_unperturbed(self):
        """
        Parallel also doing sampling of the unperturbed version
        """
        sampled_token, sampled_output, sampled_attention_mask = self._sample_tokens(self.last_logits_unperturbed, self.input_ids_unperturbed, self.attention_mask_unperturbed)
        # cur_input = sampled_token
        cur_input = sampled_output
        self.input_ids_unperturbed = sampled_output
        self.attention_mask_unperturbed = sampled_attention_mask

        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.attention_mask_unperturbed,self.past_kvs_unperturbed)
        local_logits = all_logits[:, -1, :]
        self.last_logits_unperturbed = local_logits
        self.past_kvs_unperturbed = new_past_kvs

    def step(self, action):
        """
        :param action: bool tensor of shape [batch_size]
        """

        reward = self.zero_tensor.clone().detach()
        F_GPT_Score_drop = self.zero_tensor.clone().detach().float()
        perturbed_score = self.zero_tensor.clone().detach().float()
        unperturbed_score = self.zero_tensor.clone().detach().float()
        RL_num_perturb = self.zero_tensor.clone().detach().long()

        # Parse Action
        obs, done = self._step_sample(perturb=action)
        # Also parallelly performing unperturbed samples
        self._step_sample_unperturbed()

        print("Step:", self.input_ids.shape[-1])

        if self.args.rule_based_penalty:
          rule_based_penalty = 1
        else:
          rule_based_penalty = 0

        not_done = torch.logical_not(done)

        self.num_perturb = self.num_perturb + torch.where(action & not_done, 1, 0)
        penalized_low = (action & torch.tensor(obs[:,-1, 0] <= 0.55).to(self.args.env_device)).bool()
        penalized_high = (torch.logical_not(action) & torch.tensor(obs[:,-1, 0] > 0.55).to(self.args.env_device)).bool()
        reward[penalized_low] -= rule_based_penalty
        reward[penalized_high] -= rule_based_penalty

        self.past_obs = obs

        self.sample_done = self.sample_done | done
        if self.input_ids.shape[1] >= self.max_sample_tokens:
          self.sample_done = self.one_tensor.clone().detach()
        print("Done:", self.sample_done)

        self.output_mask = torch.cat(
                    [self.output_mask,
                     torch.logical_not(self.sample_done).unsqueeze(dim=1)],
                    dim=-1)
        self.input_mask = torch.cat(
                    [self.input_mask,
                     torch.zeros_like(self.sample_done).int().unsqueeze(dim=1)],
                    dim=-1)

        if torch.all(self.sample_done):

            mask = self.input_mask | torch.logical_not(self.output_mask)
            perturbed_score = detector.infer(self.get_texts(mask))

            RL_num_perturb = self.num_perturb.clone().detach()

            unperturbed_score = detector.infer(self.get_texts_unperturbed(mask))

            F_GPT_Score_drop = 100. * (unperturbed_score - perturbed_score)

            # Reward
            reward += 100 * F_GPT_Score_drop
            reward -= 0.01 * RL_num_perturb * RL_num_perturb / 2

        info = {"TimeLimit.truncated": self.zero_tensor.clone().detach().bool().to(self.args.env_device),
                "F_GPT_Score_drop": F_GPT_Score_drop,
                "last_perturbed_score": perturbed_score,
                "last_unperturbed_score": unperturbed_score,
                "RL_num_perturb": RL_num_perturb,
                "last_reward": reward,
                }

        # If your environment does not have a concept of truncation, you can set truncated to the same value as done
        truncated = self.sample_done.bool()
        return obs, reward, self.sample_done, truncated, info

    def seed(self, seed=None):
        self._seed = seed

