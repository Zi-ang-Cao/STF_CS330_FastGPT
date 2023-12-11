
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--max_sample_tokens", default=150, type=int)

parser.add_argument("--total_timesteps", default=1, type=int)
#1.5E5

parser.add_argument("--data_items", default="127,733,55,953,469,628,793,511", type=str)

parser.add_argument("--batch_size", default=8, type=int)

parser.add_argument("--random_seed", default=42, type=int)

parser.add_argument("--dataset", default="xsum", type=str)

parser.add_argument("--n_train", default=1000, type=int)

parser.add_argument("--topK_logistics", default=10, type=int)
parser.add_argument("--perturb_ranking", default=3, type=int)
parser.add_argument("--sampling_mode", default="likelihood", type=str)
parser.add_argument("--obs_dim", default=1, type=int)
parser.add_argument("--random", action='store_true', default=False)

parser.add_argument("--model_name", default="med", type=str)
parser.add_argument("--env_device", default="cuda", type=str)

parser.add_argument("--algorithm", default="PPO", type=str)

parser.add_argument("--tb_folder", default="./tensorboard_log", type=str)

parser.add_argument("--inference", default=False, type=bool)
parser.add_argument("--save", default=True, type=bool)

parser.add_argument("--rule_based_penalty", action='store_true', default=False)

parser.add_argument("--RL_model_name", default="raw", type=str)

parser.add_argument("--retrain_from", default="", type=str)
parser.add_argument("--cross_Sentence", action='store_true', default=False)
parser.add_argument('-f')

args = parser.parse_args()

import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info


class MyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, lm_env:LMEnv, random):
        self.env = lm_env
        super().__init__(self.env.batch_size,
                         self.env.observation_space,
                         self.env.action_space)
        self.num_envs = self.env.batch_size
        obs_space = self.env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = self.env.metadata
        self.counter = 0
        self.random = random

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_without_reset(self):
        obs, self.buf_rews, terminated, truncated, buf_infos = self.env.step(
            torch.tensor(self.actions).bool().to(self.env.args.env_device)
        )
        self.buf_dones = terminated
        for i in range(self.env.batch_size):
          buf_infos_i = {}
          for k, v in buf_infos.items():
            buf_infos_i[k] = v[i]
          self.buf_infos[i] = buf_infos_i

        self._save_obs(obs)
        res = (self._obs_from_buf(),
               np.copy(self.buf_rews.cpu()),
               np.copy(self.buf_dones.bool().cpu()),
               deepcopy(self.buf_infos))
        return res

    def step_wait(self) -> VecEnvStepReturn:
        self.counter += 1
        print("VecEnv Step: ", self.counter)
        obs, self.buf_rews, terminated, truncated, buf_infos = self.env.step(
            torch.tensor(self.actions).bool().to(self.env.args.env_device)
        )

        self.buf_dones = terminated
        for i in range(self.env.batch_size):
          buf_infos_i = {}
          for k, v in buf_infos.items():
            buf_infos_i[k] = v[i]
          self.buf_infos[i] = buf_infos_i

        if torch.all(self.buf_dones):
            # save final observation where user can get it, then reset
            print("Resetting 1")
            for i in range(self.env.batch_size):
              self.buf_infos[i]["terminal_observation"] = obs[i]
            obs, self.reset_infos = self.env.reset(random=False)

            print(np.copy(self.buf_dones.bool().cpu()))
        self._save_obs(obs)
        res = (self._obs_from_buf(),
               np.copy(self.buf_rews.cpu()),
               np.copy(self.buf_dones.bool().cpu()),
               deepcopy(self.buf_infos))

        return res

    def reset(self) -> VecEnvObs:

        print("Resetting 2")
        obs, self.reset_infos = self.env.reset(seed=self._seeds,
                                               random=self.random)
        print(obs.shape)
        self._save_obs(obs)

        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        print("Close")
        self.env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        print("Get images")
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.env.batch_size]
        return [env.render()]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        print("Render")
        return super().render(mode=mode)

    def _save_obs(self, obs: VecEnvObs) -> None:
        for key in self.keys:
          for dim in range(self.env.batch_size):
            if key is None:
                self.buf_obs[key][dim] = obs[dim]
            else:
                self.buf_obs[key][dim] = obs[key][dim]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        print("get_attr ", attr_name, indices)
        return [getattr(self.env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        print("set_attr ", attr_name, value, indices)
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        print("env_method ", method_name, indices)
        return [getattr(self.env, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util
        return [env_util.is_wrapped(self.env, wrapper_class) for _ in self._get_indices(indices)]


from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env.subproc_vec_env import  SubprocVecEnv, _flatten_obs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines3.common.env_checker import check_env

def init_env_for_agent_training(args):
    env = LMEnv(args=args)
    return MyVecEnv(env, random=args.random)

############################################

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # success_rate = self.training_env.get_success_rate(window_size=100)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            # import pdb; pdb.set_trace()
            F_GPT_Score_drop = safe_mean([ep_info["F_GPT_Score_drop"] for ep_info in self.model.ep_info_buffer])
            # self.logger.record("rollout/F_GPT_Score_drop", F_GPT_Score_drop)
            self.logger.record("rollout/F_GPT_Score_drop", F_GPT_Score_drop)

            RL_num_perturb = safe_mean([ep_info["RL_num_perturb"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/RL_num_perturb", RL_num_perturb)

            last_reward = safe_mean([ep_info["last_reward"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/last_reward", last_reward)

        return True

cust_callback = TensorboardCallback()

############################################


import datetime

############################################
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

    reward_describe = f"RP_{args.rule_based_penalty}"
    prefix = ""
    if args.cross_Sentence:
        prefix += "Cross_CT_"
    else:
        prefix += "CT_"

    args.RL_model_name = f"{prefix}S_{args.data_items}_{reward_describe}_{timestamp}"

    tb_log_name = f"{args.algorithm}/{args.RL_model_name}"

    cpt_save_path = os.path.join(args.tb_folder, tb_log_name+"_1", "model_checkpoints/")

    checkpoint_callback = CheckpointCallback(save_freq=1E3, save_path=cpt_save_path)

    cust_callback = TensorboardCallback()

    ###########################################

    vec_env = init_env_for_agent_training(args=args)

    if args.algorithm=="PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1,
                    tensorboard_log=args.tb_folder)

        if args.retrain_from != "":
            print(args.retrain_from)
            model = model.load(args.retrain_from)
            model.set_env(env=vec_env)
            print("Reload model success")

        model.learn(total_timesteps=args.total_timesteps, 
                    tb_log_name=tb_log_name)
        if args.save:
            model.save(f"{args.algorithm}/{args.RL_model_name}_T_{args.total_timesteps}.pt")
    else:
        raise NotImplementedError
