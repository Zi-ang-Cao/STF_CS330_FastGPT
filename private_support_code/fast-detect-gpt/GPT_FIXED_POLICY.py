from common_helper import *



max_sample_tokens = 200
model_name = "med"
env_device = "cuda"

algorithm = "PPO"


import gymnasium as gym
# import gym
### Baseline
data = sampled_dataset("xsum", 2048, 42)

score_list = []
for i in data:
    answer = completion(i, 10)
    score = infer(answer)
    score_list.append(score)

print("Average Score: ", np.mean(score_list))
print("Max Score: ", np.max(score_list))


### RuleBased
a_list = [0.3, 0.35, 0.4]
rulebased_score_dict = {}
for a in a_list:
    rulebased_dict = {}
    score_list = []
    perturb_list = []
    for i in data:
        n_perturb = 0
        # select i_th data
        logits = get_obs(i)[-1]
        for t in range(10000):
            # get sorted topK logits
            topK_logits, topK_items = get_topK_logits(logits, 10)
            if topK_items in stop_tokens:
                break
            else:
                if softmax(topK_logits, -1) > a:
                    next_token = topK_items[2]
                    n_perturb += 1
        # get the final score for this data
        score = infer(answer)
        score_list.append(score)
        # get the number of perturbations for this data
        perturb_list.append(n_perturb)
        # save to dict
        rulebased_dict = {
            "score": score_list,
            "perturb": perturb_list
        }

    # Compute the average reward
    reward = (score_baseline - np.mean(score_list) )* 100 - np.mean(perturb_list)
    # save to dict
    rulebased_score_dict = {
        "a": a,
        "reward": reward,
        "dict": rulebased_dict
        "mean_score": np.mean(score_list),
    }

## RL





            
        
    print("Average Score: ", np.mean(score_list))
    print("Max Score: ", np.max(score_list))



class LMEnv(gym.Env):
    def __init__(self, sampling_mode: str = "likelihood", topK_logistics: int=5, dataset: str="xsum", n_train:int = 2048, 
    random_seed:int=42, obs_dim:int = 10):

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
        self.initial_text = self._get_new_input()
        self.past_kvs = None
        self.topK_logistics = topK_logistics

        self.sampling_mode = sampling_mode  # "likelihood" or "argmax"
        self.purturb_mode = "argmax"
        self.input_ids = None

        ## RL: Basic Action Space and Obs Space
        # The first integer can take values 0 or 1 (2 possibilities)
        # The second integer can take values 1 to 10 (10 possibilities)
        # self.action_space = gym.spaces.MultiDiscrete([2, self.topK_logistics])
        # self.action_space = gym.spaces.MultiDiscrete([self.topK_logistics])
        self.obs_dim = obs_dim
        self.action_space = gym.spaces.MultiDiscrete([2, self.topK_logistics])


        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, self.topK_logistics), dtype=np.float32)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(f"CS330_FastGPT_{model_name}_{env_device}/{algorithm}/OLD_Action_MLogits")

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

    def _cat_new_word(self, sampled_token):
        return torch.cat((self.input_ids, sampled_token.clone().detach().long().expand(1, 1)), dim=1)    
    
    def _sample_tokens(self, local_logits):
        # Change 2: Return the new token as well as concatenated previous tokens
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :return new_token: works together with past_kvs returned from get_logits() to feed in the next round of get_logits().
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
        if sampled_token[0] in self.ignore_tokens:
            sampled_token[0] = self.ignore_tokens_replace

        new_token = sampled_token.unsqueeze(0)
        new_input_ids = self._cat_new_word(new_token)
        return new_token, new_input_ids
         
    def _perturb_tokens(self, local_logits, perturb_ranking):
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :param perturb_ranking: perturb selection of the last word
        :return new_token: the selected token to generate
        :return new_input_ids: the new input ids after the perturbation
        """
        # Get the top k predictions （1-10）
        _, topk_indices = torch.topk(local_logits, perturb_ranking)
        # Select the last item
        new_token = topk_indices[0][-1]
        new_input_ids = self._cat_new_word(new_token)
        return new_token, new_input_ids

    def _obs_wrapper(self, all_logits):
        # Sorted topk_values
        # TODO(ziangcao): add previous model parts to the observation
        topk_values, _ = torch.topk(all_logits, self.topK_logistics, dim=-1)
        # Normalize the topk_values
        topk_values = F.softmax(topk_values, dim=-1)
        # Remove batch dim
        topk_values = topk_values.squeeze(dim=0)
        return topk_values.detach().cpu().numpy()

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

    def _get_new_input(self):
        return self.data[np.random.randint(self.n_train)].replace('\n', ' ')
    
    def _sample_done(self):
        a = self.input_ids[0][-1] in self.stop_tokens
        b = self.input_ids.shape[1] >= self.max_sample_tokens
        return a or b
    
    def reset(self, seed: int = None):
        print("Resetting environment=============")
        ## Get a new generate starting point
        initial_text = self._get_new_input()
        self.input_ids = self.tok(initial_text, return_tensors="pt")["input_ids"].to(env_device)
        while self.input_ids.shape[-1] ==0:
            initial_text = self._get_new_input()
            self.input_ids = self.tok(initial_text, return_tensors="pt")["input_ids"].to(env_device)
        ## First 1 step
        all_logits, new_past_kvs = self._feedforward(self.input_ids)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        _, new_input_ids = self._sample_tokens(local_logits)
        self.input_ids = new_input_ids

        obs = self._obs_wrapper(all_logits)

        # reset_info = None  # or reset_info = {} if you prefer
        reset_info = {"TimeLimit.truncated": False,}  # or reset_info = {} if you prefer
        return obs, reset_info
        # return obs

    def get_text(self):
        return self.tok.decode(torch.squeeze(self.input_ids, dim=0))

    def step(self, action):
        reward = 0.
        # Parse Action
        perturb = action[0]
        ## perturb_ranking: 10 options -- shift the choice from 0-9 toward 1-10
        perturb_ranking = action[1] + 1

        # ## perturb_ranking: 10 options -- shift the choice from 0-9 toward 1-10
        # perturb_ranking = action[1] + 1

        sampled_token, sampled_output = self._sample_tokens(self.last_logits)

        if not perturb:
            self.input_ids = sampled_output
            cur_input = sampled_token
            # print("Raw: ", reward)
            prob_drop = 0.
            sampled_score = 0.
            perturbed_score = 0.
        else:
            reward -= 1. # Cost of applying perturb
            # TODO(ziangcao): better give a large value instead of 1
            _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_ranking)

            # Record Scores -- prob
            print()
            sampled_score = detector.infer(self.tok.decode(torch.squeeze(sampled_output, dim=0)))
            perturbed_score = detector.infer(self.tok.decode(torch.squeeze(perturbed_output, dim=0)))

            assert sampled_score>=0
            assert perturbed_score>=0

            reward += (sampled_score-perturbed_score) * 100. # Benefits of applying perturb

            self.input_ids = perturbed_output
            cur_input = self.input_ids
            self.past_kvs = None

            # print("Perturbed: ", reward)
        

        idx = self.input_ids.shape[1]
        prob_drop = sampled_score-perturbed_score

        self.writer.add_scalar("perturb", perturb, idx)
        self.writer.add_scalar("reward", reward, idx)
        self.writer.add_scalar("Prob_Drop", prob_drop, idx)
        self.writer.add_scalar("sampled_score", sampled_score, idx)
        self.writer.add_scalar("perturbed_score", perturbed_score, idx)


        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.past_kvs)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        obs = self._obs_wrapper(all_logits)

        info = {"TimeLimit.truncated": False,}

        done = self._sample_done()

        # If your environment does not have a concept of truncation, you can set truncated to the same value as done
        truncated = done
        return obs, reward, done, truncated, info
        # return obs, reward, done, info

    
    def seed(self, seed=None):
        self._seed = seed
        pass
    



from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env.subproc_vec_env import  SubprocVecEnv, _flatten_obs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines3.common.env_checker import check_env

def init_env_for_agent_training(n_envs: int=1):
    def make_env():
        def _make_env():
            env=LMEnv()
            check_env(env)

            return env
        
        if n_envs == -1:
            return _make_env()
        else:
            return _make_env()

    if n_envs == -1:
        return make_env()
    if n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

############################################

vec_env = init_env_for_agent_training()

if algorithm=="PPO":
    model = PPO("MlpPolicy", vec_env, verbose=1, 
                tensorboard_log="./tensorboard_log")
    model.learn(total_timesteps=2E5, tb_log_name=f"{algorithm}/old_ActionSpace_MLogits")
    # model.save("FirstAgent")
