from common_helper_batch import *

from stable_baselines3.common.monitor import Monitor



max_sample_tokens = 200
model_name = "med"
env_device = "cuda"

algorithm = "PPO"
# If true, infer once with the model
inference = False
# If true, save the model
save = False
# If true, use both reward. Otherwise, only rule reward.
both_reward = True

rulePenaltyOnly = True
# Number of environments to simulate
n_envs = 4

# Tensorboard log names
RL_model_name = "fixedACT_BATCH"
# "fixedACT_MONITOR_onlyRuleReward"


import gymnasium as gym
import tqdm
# import gym



class LMEnv(gym.Env):
    ### NOTE: [CHANGE!!!] change the n_train from 8 to 1
    ### NOTE: [CHANGE!!!] change the sampling_mode from "likelihood" to "argmax"
    def __init__(self, sampling_mode: str = "argmax", topK_logistics: int=10, dataset: str="xsum", n_train:int = 1,
    random_seed:int=42, obs_dim:int = 1):

        # Dataset
        self.random_seed = random_seed
        self.dataset = dataset
        self.n_train = n_train
        self._load_datasets()

        ## LLM
        self.max_sample_tokens = max_sample_tokens
        self.model, self.tok = get_model_and_tokenizer(model_name)
        assert isinstance(self.model, transformers.GPT2LMHeadModel)
        self.model.to(env_device)
        self.stop_tokens = stop_tokens(self.tok)
        self.ignore_tokens = ignore_tokens(self.tok)
        self.ignore_tokens_replace = ignore_tokens_replace(self.tok)
        self._seed = None
        self.vocab_size = len(self.tok)
        # Current inputs and logits
        
        self.topK_logistics = topK_logistics
        self.batch_size = 10

        self.sampling_mode = sampling_mode  # "likelihood" or "argmax"
        self.data_items = None
        self.num_perturb = None
        self.past_obs = None
        
        self.input_ids = None
        self.past_kvs = None
        self.last_logits = None
        self.last_logits_unperturbed = None
        self.input_ids_unperturbed = None
        self.past_kvs_unperturbed = None

        ## RL: Basic Action Space and Obs Space
        # Whether perturb or not.
        # If not perturb: sample by multinomial
        # If perturb: sample by equal probability
        self.obs_dim = obs_dim
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, self.topK_logistics), dtype=np.float32)

        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(f"CS330_FastGPT_{model_name}_{env_device}/{algorithm}/OLD_Action_MLogits")
        self.zero_tensor = torch.zeros(self.batch_size).to(env_device)
        self.one_tensor = torch.ones(self.batch_size).to(env_device)

        self.reset()

    def _feedforward(self, cur_input, past_kvs=None):
        # Change 1: Speed up feedforward by utilizing past_kvs
        """
        :param cur_input: When past_kvs = None, tensor shape [batch_size, seq_len]. When past_kvs is not None, tensor shape [batch_size, 1]
        :param past_kvs: a cache to speed up model inference
        :return local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :return new_past_kvs: the new model state cache
        """
        # print("cur_input: ", cur_input.shape)
        # if cur_input.shape[-1] ==0:
        #     input()
        with torch.inference_mode():
            outputs = self.model(cur_input, past_key_values=past_kvs, use_cache=True)
            all_logits = outputs.logits
            B, S, V = all_logits.shape
            returned_logits = torch.ones(B, self.obs_dim, V).float().to(env_device)
            if S < self.obs_dim:
                returned_logits[:, self.obs_dim - S:, :] = all_logits
            else:
                returned_logits = all_logits[:, S - self.obs_dim:, :]
            new_past_kvs = outputs.past_key_values
            return returned_logits, new_past_kvs

    def _cat_new_word(self, sampled_token, input_ids):
        token_len = sampled_token.shape[0]
        return torch.cat((input_ids, sampled_token.clone().detach().long().view(-1, 1)), dim=1)

    def _sample_tokens(self, local_logits, input_ids):
        # Change 2: Return the new token as well as concatenated previous tokens
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :param input_ids: tensor shape [batch_size, seq_len] input ids at latest point
        :return new_token: tensor shape [batch_size, 1]
        works together with past_kvs returned from get_logits() to feed in the next round of get_logits().
        :return new_input_ids: when past_kvs = None, this would return the complete input concat with output up to this point
        """
        if self.sampling_mode == "argmax":
            sampled_token = torch.argmax(local_logits, dim=-1)
        elif self.sampling_mode == "likelihood":
            # print(local_logits.shape, x.shape)
            sampled_token = torch.multinomial(F.softmax(local_logits, dim=-1), num_samples=1).squeeze(dim=1)
            # sampled_token = torch.multinomial(x, num_samples=1).squeeze(dim=1)
        else:
            raise NotImplementedError

        # Replace tokens such as new line with spaces

        mask = torch.any(torch.eq(sampled_token, torch.tensor(self.ignore_tokens).to(device)), dim=-1)
        sampled_token[mask] = self.ignore_tokens_replace

        new_token = sampled_token.view(-1, 1)
        new_input_ids = self._cat_new_word(new_token, input_ids)
        return new_token, new_input_ids

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
            new_input_ids = self._cat_new_word(new_token)
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

    def _load_datasets(self):
        print("Dataset:", self.dataset)
        if self.dataset == "xsum":
            d = datasets.load_dataset(self.dataset, split="train").shuffle(seed=self.random_seed)
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

    def _sample_done(self):
        input_ids = self.input_ids[:,-1].unsqueeze(dim=-1)
        stop_tokens = torch.tensor(self.stop_tokens).view(1, -1).to(env_device)
        # print("Token:", token.shape, "Stop Tokens:", stop_tokens.shape)
        # (batch_size=1, topk=10) & (num_stop_tokens=2)

        a = torch.any(torch.eq(input_ids, stop_tokens))
        print("Done:", a, input_ids[0][0], stop_tokens)
        b = self.input_ids.shape[1] >= self.max_sample_tokens
        b = torch.tensor(b).repeat((self.batch_size)).to(env_device)
        return a | b

    def _masked(self, a, b, mask):
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

    def _reset(self, random=True, data_items=None, mask=None):
        """
        :param random: whether to sample data randomly
        :param data_items: if random == False, choose these data items in the dataset
        :param mask: only reset these rows in the batch
        """
        print("Reset begins...")
        if mask is None:
            mask = self.one_tensor.long().detach()
        if not torch.any(mask):
            return self.past_obs, None
        if random:
            data_items = torch.randint(low=0, high=self.n_train, size=(self.batch_size,)).to(env_device)
        self.data_items = self._masked(self.data_items, data_items, mask)
        print("Data Items:", self.data_items)
        
        ## Get a new generate starting point
        initial_texts = self._get_new_input(self.data_items)

        self.input_ids = self._masked(self.input_ids,
                     self.tok(initial_texts, 
                              return_tensors="pt")["input_ids"]
                     .to(env_device),
                     mask)
        self.num_perturb = self._masked(self.num_perturb, self.zero_tensor.detach(), mask)
        while random and self.input_ids.shape[-1] ==0:
            data_items = torch.randint(low=0, high=self.n_train, size=(self.batch_size,)).to(env_device)
            self.data_items = self._masked(self.data_items,
                         data_items,
                         mask)
            initial_texts = self._get_new_input(self.data_items)
            self.input_ids = self._masked(self.input_ids,
                         self.tok(initial_texts, 
                                  return_tensors="pt")["input_ids"]
                         .to(env_device),
                         mask)
        ## First 1 step
        all_logits, new_past_kvs = self._feedforward(self.input_ids)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        _, new_input_ids = self._sample_tokens(local_logits, self.input_ids)
        self.input_ids = new_input_ids

        self.last_logits_unperturbed = self.last_logits
        self.past_kvs_unperturbed = self.past_kvs
        self.input_ids_unperturbed = self.input_ids

        
        obs, _ = self._obs_wrapper(all_logits)

        ## NOTE: save the past obs
        self.past_obs = obs

        reset_info = {"TimeLimit.truncated": self.zero_tensor.detach().bool(),
                      "DataItem": self.data_items,
                      "F_GPT_Score_drop": self.zero_tensor.detach(),
                      "RL_num_perturb": self.zero_tensor.detach(),
                      "last_reward": self.zero_tensor.detach(),
                      }
        print("Reset ends!")
        print(obs, reset_info)
        return obs, reset_info

    def reset(self, seed: int = None, mask = None):
        # print("Resetting environment=============")
        return self._reset(random=True, data_items=None, mask=mask)
        # return obs

    def get_texts(self):
        """
        :return texts: str list [batch_size]
        """
        # print(self.input_ids.shape)
        return self.tok.batch_decode(self.input_ids)

    def get_texts_unperturbed(self):
        """
        :return texts: str list [batch_size]
        """
        # print(self.input_ids_unperturbed.shape)
        return self.tok.batch_decode(self.input_ids_unperturbed)

    def _step_sample(self, perturb):
        """
        :param perturb: boolean tensor of shape [batch_size]
        :return obs: tensor of shape [batch_size, obs_dim, topk]
        :return done: bool tensor of shape [batch_size]
        """
        sampled_token, sampled_output = self._sample_tokens(self.last_logits, self.input_ids)

        _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_mode="chosen", perturb_ranking=3)
        print(perturb.device, perturbed_output.device, sampled_output.device)
        self.input_ids = torch.where(perturb.unsqueeze(dim=-1), perturbed_output, sampled_output)
        if torch.any(perturb):
            cur_input = self.input_ids
            self.past_kvs = None
        else:
            cur_input = sampled_token

        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.past_kvs)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        obs, token = self._obs_wrapper(all_logits)
        token = torch.tensor(token)[:, -1, :].unsqueeze(dim=-1).to(env_device)
        stop_tokens = torch.tensor(self.stop_tokens).view(1, 1, -1).to(env_device)
        # print("Token:", token.shape, "Stop Tokens:", stop_tokens.shape)
        # (batch_size=1, topk=10) & (num_stop_tokens=2)

        done = torch.any(torch.eq(token, stop_tokens), dim=-1)
        done = torch.any(done, dim=-1)
        print("Done?", done, token[0, 0, 0])

        return obs, done

    def _step_sample_unperturbed(self):
        """
        Parallel also doing sampling of the unperturbed version
        """
        sampled_token, sampled_output = self._sample_tokens(self.last_logits_unperturbed, self.input_ids_unperturbed)
        cur_input = sampled_token
        self.input_ids_unperturbed = sampled_output
        print("Perturbed!")

        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.past_kvs_unperturbed)
        local_logits = all_logits[:, -1, :]
        self.last_logits_unperturbed = local_logits
        self.past_kvs_unperturbed = new_past_kvs

    def step(self, action):
        """
        :param action: bool tensor of shape [batch_size]
        """
        reward = self.zero_tensor.detach()
        F_GPT_Score_drop = self.zero_tensor.detach().float()
        RL_num_perturb = self.zero_tensor.detach().long()

        # Parse Action
        obs, done = self._step_sample(perturb=action)
        # Also parallelly performing unperturbed samples
        self._step_sample_unperturbed()

        if rulePenaltyOnly:
            Upper_threshold = 0.7
            gain = 0
        else:
            Upper_threshold = 0.55
            gain = 1

        not_done = torch.logical_not(done)
        # print(action.shape, not_done.shape)
        self.num_perturb += torch.where(action & not_done, 1, 0)
        reward += torch.where(done, 0, torch.where(action, -1, gain))

        ## NOTE: save the past obs
        self.past_obs = obs

        done = done | self._sample_done()

        fake_reward = reward

        if torch.any(done):
            print("Done!!!")
            perturbed_score = detector.infer(self.get_texts())

            RL_num_perturb = self.num_perturb

            unperturbed_score = detector.infer(self.get_texts_unperturbed())

            F_GPT_Score_drop = 100. * (unperturbed_score - perturbed_score)

            # NOTE: Reward
            fake_reward += 10 * F_GPT_Score_drop
            fake_reward -= 0.01 * RL_num_perturb * RL_num_perturb / 2

            F_GPT_Score_drop[~done] = 0.
            RL_num_perturb[~done] = 0
            fake_reward[~done] = 0.

        info = {"TimeLimit.truncated": False,
                "F_GPT_Score_drop": F_GPT_Score_drop,
                "RL_num_perturb": RL_num_perturb,
                "last_reward": fake_reward,
                }

        if both_reward:
            reward = fake_reward

        # If your environment does not have a concept of truncation, you can set truncated to the same value as done
        truncated = done
        return obs, reward, done, truncated, info
        # return obs, reward, done, info


    def seed(self, seed=None):
        self._seed = seed

def manual_policy(env: LMEnv, threshold = 0.55, num_samples = 100):
    rewards = []

    pbar = tqdm.tqdm(range(num_samples))
    for _ in pbar:
        done = False
        num_perturb = 0
        tot = 0
        reward = 0.
        reset_mask = None
        
        obs, _ = env.reset(mask=reset_mask)
        
        mask = obs[:, 0] > threshold
        mask = torch.tensor(mask)[:, 0].bool().to(env_device)
        action = torch.where(mask, 1, 0).bool().to(env_device)
        print("Action:", action)
        num_perturb += torch.where(mask, 1, 0)
        obs, local_reward, local_done, _, _ = env.step(action)
        reset_mask = local_done
        reward += local_reward

        pbar.set_description(f"Reward: {reward}")
        rewards.append(reward.mean().cpu())
    print("Rewards Mean: ", np.mean(rewards), "Std: ", np.std(rewards))

