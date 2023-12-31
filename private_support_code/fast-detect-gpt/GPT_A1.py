from common_helper import *

from stable_baselines3.common.monitor import Monitor

import datetime
import gymnasium as gym
import tqdm
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env.subproc_vec_env import  SubprocVecEnv, _flatten_obs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines3.common.env_checker import check_env


class LMEnv(gym.Env):
    def __init__(self, args, sampling_mode: str = "argmax", topK_logistics: int=10, dataset: str="xsum", n_train:int = 1, 
    random_seed:int=42, obs_dim:int = 1, overfit_sentence_idx=-1):
        

        self.args=args
        if overfit_sentence_idx == -1:
            self.overfit_sentence_idx = self.args.overfit_sentence_idx
        else:
            self.overfit_sentence_idx = overfit_sentence_idx

        # Dataset
        self.random_seed = random_seed
        self.dataset = dataset
        self.n_train = n_train
        self._load_datasets()

        ## LLM
        self.max_sample_tokens = self.args.max_sample_tokens
        self.model, self.tok = get_model_and_tokenizer(self.args.model_name)
        assert isinstance(self.model, transformers.GPT2LMHeadModel)
        self.model.to(self.args.env_device)
        self.stop_tokens = stop_tokens(self.tok)
        self.ignore_tokens = ignore_tokens(self.tok)
        self.ignore_tokens_replace = ignore_tokens_replace(self.tok)
        self._seed = None
        self.vocab_size = len(self.tok)
        # Current inputs and logits
        self.past_kvs = None
        self.topK_logistics = topK_logistics

        self.sampling_mode = sampling_mode  # "likelihood" or "argmax"
        self.input_ids = None

        ## RL: Basic Action Space and Obs Space
        # Whether perturb or not.
        # If not perturb: sample by multinomial
        # If perturb: sample by equal probability
        self.obs_dim = obs_dim
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, self.topK_logistics), dtype=np.float32)

        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(f"CS330_FastGPT_{model_name}_{env_device}/{algorithm}/OLD_Action_MLogits")

        self.reset()

    
    def _feedforward(self, cur_input, past_kvs=None):
        """
        :param cur_input: When past_kvs = None, tensor shape [batch_size, seq_len]. When past_kvs is not None, tensor shape [batch_size, 1]
        :param past_kvs: a cache to speed up model inference
        :return local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :return new_past_kvs: the new model state cache
        """
        with torch.inference_mode():
            outputs = self.model(cur_input, past_key_values=past_kvs, use_cache=True)
            all_logits = outputs.logits
            B, S, V = all_logits.shape
            returned_logits = torch.ones(B, self.obs_dim, V).float().to(self.args.env_device)
            if S < self.obs_dim:
                returned_logits[:, self.obs_dim - S:, :] = all_logits
            else:
                returned_logits = all_logits[:, S - self.obs_dim:, :]
            new_past_kvs = outputs.past_key_values
            return returned_logits, new_past_kvs

    def _cat_new_word(self, sampled_token):
        return torch.cat((self.input_ids, sampled_token.clone().detach().long().expand(1, 1)), dim=1)    
    
    def _sample_tokens(self, local_logits):
        """
        :param local_logits: tensor shape [batch_size, vocab_size] local logits at the last point
        :return new_token: works together with past_kvs returned from get_logits() to feed in the next round of get_logits().
        :return new_input_ids: when past_kvs = None, this would return the complete input concat with output up to this point
        """
        if self.sampling_mode == "argmax":
            sampled_token = torch.argmax(local_logits, dim=-1)
        elif self.sampling_mode == "likelihood":
            sampled_token = torch.multinomial(F.softmax(local_logits, dim=-1), num_samples=1).squeeze(dim=1)
        else:
            raise NotImplementedError
        
        # Replace tokens such as new line with spaces
        if sampled_token[0] in self.ignore_tokens:
            sampled_token[0] = self.ignore_tokens_replace

        new_token = sampled_token.unsqueeze(0)
        new_input_ids = self._cat_new_word(new_token)
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
            _, topk_indices = torch.topk(local_logits, perturb_ranking)
            # Select the last item
            new_token = topk_indices[0][-1]
            new_input_ids = self._cat_new_word(new_token)
            return new_token, new_input_ids
        else:
            _, topk_indices = torch.topk(local_logits, 10)
            # Select random item
            new_token = topk_indices[0][random.randint(0, 9)]
            new_input_ids = self._cat_new_word(new_token)
            return new_token, new_input_ids

    def _obs_wrapper(self, all_logits):
        # Sorted topk_values
        ## Detect End of Sentence
        topk_values, topk_indices = torch.topk(all_logits, self.topK_logistics, dim=-1)
        # Normalize the topk_values
        topk_values = F.softmax(topk_values, dim=-1)
        # Remove batch dim
        topk_values = topk_values.squeeze(dim=0)

        ## Check the first token is in stop_tokens
        topk_indices = topk_indices.squeeze(dim=0)

        return topk_values.detach().cpu().numpy(), topk_indices.detach().cpu().numpy()

    def _load_datasets(self):
        print("Dataset:", self.dataset)
        if self.dataset == "xsum":
            d = datasets.load_dataset(self.dataset, split="train").shuffle(seed=self.random_seed)
            filter_fn = lambda rows: [
                len(a.split(" ")) < 100 for a in rows["document"]
            ]
            d = d.filter(filter_fn, batched=True, batch_size=None)

            start_idx = self.overfit_sentence_idx
            print(f">>>>>>>>> FOR THIS ENV, start_idx={start_idx}")
            end_idx = start_idx + self.n_train
            d = d["document"][start_idx:end_idx]
            self.data = d
        else:
            raise NotImplementedError

    def _get_new_input(self, item):
        return self.data[item].replace('\n', ' ')
    
    def _sample_done(self):
        a = self.input_ids[0][-1] in self.stop_tokens
        b = self.input_ids.shape[1] >= self.max_sample_tokens
        return a or b
    
    def _reset(self, random=True, data_item=-1):
        if random:
            self.data_item = np.random.randint(self.n_train)
        else:
            self.data_item = data_item
        ## Get a new generate starting point
        initial_text = self._get_new_input(self.data_item)
        self.input_ids = self.tok(initial_text, return_tensors="pt")["input_ids"].to(self.args.env_device)
        self.num_perturb = 0
        while random and self.input_ids.shape[-1] ==0:
            self.data_item = np.random.randint(self.n_train)
            initial_text = self._get_new_input(self.data_item)
            self.input_ids = self.tok(initial_text, return_tensors="pt")["input_ids"].to(self.args.env_device)
        
        ## First 1 step
        all_logits, new_past_kvs = self._feedforward(self.input_ids)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        _, new_input_ids = self._sample_tokens(local_logits)
        self.input_ids = new_input_ids
        self.initial_input_ids = self.input_ids

        # NEW
        self.input_ids_noPerturb = new_input_ids

        obs, _ = self._obs_wrapper(all_logits)

        ## save the past obs
        self.past_obs = obs

        reset_info = {"TimeLimit.truncated": False,
                      "DataItem": self.data_item, 
                      "F_GPT_Score_drop": 0,
                      "RL_num_perturb": 0,
                      "last_reward": 0,
                      }  
        return obs, reset_info
    
    def reset(self, seed: int = None):
        # print("Resetting environment=============")
        return self._reset(random=True, data_item=-1)
        # return obs

    def get_text(self):
        return self.tok.decode(torch.squeeze(self.input_ids, dim=0))

    def get_text_noPerturb(self):
        return self.tok.decode(torch.squeeze(self.input_ids_noPerturb, dim=0))
    
    def _step_sample(self, perturb):
        sampled_token, sampled_output = self._sample_tokens(self.last_logits)
        # NEW
        self.input_ids_noPerturb = sampled_output

        if not perturb:
            self.input_ids = sampled_output
            cur_input = sampled_token
        else:
            # _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_mode="random")

            # Change the perturb mode to chosen
            _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_mode="chosen", perturb_ranking=3)


            self.input_ids = perturbed_output
            cur_input = self.input_ids
            self.past_kvs = None

        ## GET NEW OBS
        all_logits, new_past_kvs = self._feedforward(cur_input, self.past_kvs)
        local_logits = all_logits[:, -1, :]
        self.last_logits = local_logits
        self.past_kvs = new_past_kvs

        obs, token = self._obs_wrapper(all_logits)
        if token[-1][0] in self.stop_tokens:
            self._cat_new_word(torch.tensor(token[-1][0]).to(self.args.env_device))
            return obs, True
        else:
            return obs, False

    def step(self, action):
        reward = 0.
        F_GPT_Score_drop = 0.
        RL_num_perturb = 0
        perturbed_score = 0
        unperturbed_score =0

        # Parse Action
        obs, done = self._step_sample(perturb=action)

        # With RuleBased heuristic
        if self.args.ruleBasedPenalty:
            ruleBasedPenalty = 1
        else:
            ruleBasedPenalty = 0

        
        if not done:
            if action:
                self.num_perturb += 1

                if ruleBasedPenalty != 0:
                    if self.past_obs[-1][0] <= 0.55:
                        reward -= ruleBasedPenalty
                
                ### NEW ###
                if self.args.InStepScore:
                    perturbed_score = detector.infer(self.get_text())
                    unperturbed_score = detector.infer(self.get_text_noPerturb())
                    F_GPT_Score_drop = 100. * (unperturbed_score - perturbed_score)
                    reward += 1 * F_GPT_Score_drop

                    if self.args.InStepPerturbCost:
                        reward -= 0.1

            else:
                if ruleBasedPenalty != 0:
                    if self.past_obs[-1][0] > 0.55:
                        reward -= ruleBasedPenalty

        
        ## save the past obs
        self.past_obs = obs

        done = done or self._sample_done()

        last_reward = 0

        if done:
            perturbed_score = detector.infer(self.get_text())

            RL_num_perturb = self.num_perturb

            self._reset(random=False, data_item=self.data_item)
            while not self._sample_done():
                self._step_sample(perturb=False)
            
            unperturbed_score = detector.infer(self.get_text())

            F_GPT_Score_drop = 100. * (unperturbed_score - perturbed_score)

            # Reward
            last_reward += 10 * F_GPT_Score_drop
            last_reward -= 0.01 * RL_num_perturb * RL_num_perturb / 2
        #     Negative reward grows by O(N^2)
        
        reward += last_reward

        info = {"TimeLimit.truncated": False, 
                "F_GPT_Score_drop": F_GPT_Score_drop,
                "last_perturbed_score": perturbed_score,
                "last_unperturbed_score": unperturbed_score,
                "RL_num_perturb": RL_num_perturb,
                "last_reward": last_reward,
                }

        # If your environment does not have a concept of truncation, you can set truncated to the same value as done
        truncated = done
        
        return obs, reward, done, truncated, info

    
    def seed(self, seed=None):
        self._seed = seed
    
def manual_policy(env: LMEnv, threshold = 0.55, chosen = 2, num_samples = 100):
    rewards = []

    pbar = tqdm.tqdm(range(num_samples))
    for _ in pbar: 
        num_perturb = 0
        tot = 0
        reward = 0.
        while tot == 0:
            obs, _ = env.reset()
            while not env._sample_done():
                tot += 1
                if obs[-1][0] > threshold:
                    num_perturb += 1
                    obs, local_reward, _, _, _ = env.step((True, chosen))
                else:
                    obs, local_reward, _, _, _ = env.step((False, -1))
                reward += local_reward
        
        pbar.set_description(f"Reward: {reward:.04f}")
        rewards.append(reward)
    print("Rewards Mean: ", np.mean(rewards), "Std: ", np.std(rewards))

def train_manual_policy():
    env = LMEnv(sampling_mode="likelihood")
    manual_policy(env)

def default_env_maker(args, n_envs: int=1):
    def make_env():
        def _make_env(args):
            env=LMEnv(args)
            check_env(env)
            return env
        
        if n_envs == -1:
            return _make_env(args)
        else:
            return CustomMonitor(_make_env(args))

    if n_envs == -1:
        return make_env()
    if n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

def init_env_for_agent_training(args, n_envs: int=1):
    if n_envs==1:
        return default_env_maker(args)
    elif n_envs>1:
        if n_envs == 2:
            def sepcfic_env_0():
                return CustomMonitor(LMEnv(args, overfit_sentence_idx=0))
            def sepcfic_env_1():
                return CustomMonitor(LMEnv(args, overfit_sentence_idx=1))
            return SubprocVecEnv([sepcfic_env_0, sepcfic_env_1])
        else:
            raise NotImplementedError


class CustomMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        obs, reward, done, truncated, info = super().step(action)

        if done:
            info["episode"]["F_GPT_Score_drop"] = info.get('F_GPT_Score_drop')
            info["episode"]["RL_num_perturb"] = info.get('RL_num_perturb')
            info["episode"]["last_reward"] = info.get('last_reward')

            info["episode"]["last_perturbed_score"] = info.get('last_perturbed_score')
            info["episode"]["last_unperturbed_score"] = info.get('last_unperturbed_score')

        return obs, reward, done, truncated, info

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            # import pdb; pdb.set_trace()
            F_GPT_Score_drop = safe_mean([ep_info["F_GPT_Score_drop"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/F_GPT_Score_drop", F_GPT_Score_drop)

            RL_num_perturb = safe_mean([ep_info["RL_num_perturb"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/RL_num_perturb", RL_num_perturb)

            last_reward = safe_mean([ep_info["last_reward"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/last_reward", last_reward)

            last_perturbed_score = safe_mean([ep_info["last_perturbed_score"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/last_perturbed_score", last_perturbed_score)

            last_unperturbed_score = safe_mean([ep_info["last_unperturbed_score"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/last_unperturbed_score", last_unperturbed_score)

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--max_sample_tokens", default=200, type=int)

    parser.add_argument("--total_timesteps", default=1.5E5, type=int)

    parser.add_argument("--overfit_sentence_idx", default=0, type=int)

    parser.add_argument("--model_name", default="med", type=str)
    parser.add_argument("--env_device", default="cuda", type=str)

    parser.add_argument("--algorithm", default="PPO", type=str)

    parser.add_argument("--tb_folder", default="./tensorboard_log", type=str)

    parser.add_argument("--inference", default=True, type=bool)
    parser.add_argument("--save", default=True, type=bool)


    parser.add_argument("--ruleBasedPenalty", action='store_true', default=False)
    parser.add_argument("--InStepScore", action='store_true', default=False)
    parser.add_argument("--InStepPerturbCost", action='store_true', default=False)


    parser.add_argument("--RL_model_name", default="raw", type=str)

    parser.add_argument("--retrain_from", default="", type=str)
    parser.add_argument("--cross_Sentence", action='store_true', default=False)

    parser.add_argument("--num_env", default=1, type=int)


    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

    reward_describe = f"RP_{args.ruleBasedPenalty}_SC_{args.InStepScore}_SPC_{args.InStepPerturbCost}"

    print(reward_describe)

    prefix = ""

    if args.num_env > 1: 
        prefix = f"Envs_{args.num_env}_"
        args.overfit_sentence_idx=0
        print("reset overfit_sentence_idx=0")

    # exit(0)
    if args.cross_Sentence:
        prefix += "Cross_CT_"
    else:
        prefix += "CT_"

    args.RL_model_name = f"{prefix}S_{args.overfit_sentence_idx}_{reward_describe}_{timestamp}"

    tb_log_name = f"{args.algorithm}/{args.RL_model_name}"

    cpt_save_path = os.path.join(args.tb_folder, tb_log_name+"_1", "model_checkpoints/")

    checkpoint_callback = CheckpointCallback(save_freq=5E3, save_path=cpt_save_path)

    cust_callback = TensorboardCallback()

    ############################################

    vec_env = init_env_for_agent_training(args, n_envs=args.num_env)


    if args.algorithm=="PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1, 
                    tensorboard_log=args.tb_folder)
        
        if args.retrain_from != "":
            print(args.retrain_from)
            model = model.load(args.retrain_from)
            print("success!!!")

        model.set_env(env=vec_env)

        model.learn(total_timesteps=args.total_timesteps, tb_log_name=tb_log_name, 
                    callback=[cust_callback, checkpoint_callback])
                    # )
        if args.save:
            model.save(f"{args.algorithm}/{args.RL_model_name}_T_{args.total_timesteps}.pt")
        

        if args.num_env > 1:
            print("Need to early exit --- the current inference is not support for multiple envs")
            exit(0)

        if args.inference:
            if args.save:
                model = PPO.load(f"{args.algorithm}/{args.RL_model_name}_T_{args.total_timesteps}.pt")

            #### RL Policy + GPT2 ####
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = vec_env.step(action)
            print("RL Policy + GPT2: ", vec_env.env_method("get_text"))

    ### DQN ###
    elif args.algorithm=="DQN":
        model = DQN("MlpPolicy", vec_env, verbose=1, 
                    tensorboard_log=args.tb_folder)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=tb_log_name, 
                    callback=[cust_callback, checkpoint_callback])
                    # )
        if args.save:
            model.save(f"{args.algorithm}/{args.RL_model_name}_T_{args.total_timesteps}.pt")
        
        if args.inference:
            if args.save:
                model = DQN.load(f"{args.algorithm}/{args.RL_model_name}_T_{args.total_timesteps}.pt")

            #### RL Policy + GPT2 ####
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = vec_env.step(action)
            print("RL Policy + GPT2: ", vec_env.env_method("get_text"))
