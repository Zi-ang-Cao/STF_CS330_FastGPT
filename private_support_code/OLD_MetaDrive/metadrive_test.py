from MetaDrive_Env import MetaDriveEnv
import argparse
import numpy as np
from mat.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

parser = argparse.ArgumentParser(description='metadrive')
args = parser.parse_args()
args.scenario_name = 'roundabout'
args.use_render = False
args.num_agents = 3
args.n_rollout_threads = 4
args.seed = 0


print('test the original environment...')
env = MetaDriveEnv(args)
action_dim = env.action_space[0].shape[0]
env.reset()
for i in range(10):
    action = {'agent0': np.zeros((action_dim)), 'agent1': np.zeros((action_dim)), 'agent2': np.zeros((action_dim))}
    obs, reward, done, info = env.step(action)
    # print('done', done)
env.close()


print('test the vectorized environment...')
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MetaDriveEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    
env = make_train_env(args)
obs = env.reset()
# print('obs0', obs.shape)
for i in range(100):
    # action = [{'agent0': np.zeros((action_dim)), 'agent1': np.zeros((action_dim)), 'agent2': np.zeros((action_dim))},
    #           {'agent0': np.zeros((action_dim)), 'agent1': np.zeros((action_dim)), 'agent2': np.zeros((action_dim))},
    #           {'agent0': np.zeros((action_dim)), 'agent1': np.zeros((action_dim)), 'agent2': np.zeros((action_dim))},
    #           {'agent0': np.zeros((action_dim)), 'agent1': np.zeros((action_dim)), 'agent2': np.zeros((action_dim))}]
    action = np.zeros((args.n_rollout_threads, args.num_agents, action_dim))
    obs, reward, done, info = env.step(action)
    # print('done', done.shape)
    # print('obs1', obs.shape)
    # print('reward1', reward.shape)
env.close()