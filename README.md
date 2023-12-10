# STF_CS330_FastGPT

## Setup
```Shell
# Download the latest Linux installer via wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh

# Execute .sh 
bash Miniconda3-latest-Linux-x86_64.sh
# After installation, open a new terminal or run `source ~/.bashrc` to activate the conda env.

# Change the install path to /home/ziangcao2022/workspace/miniconda3
```

```Shell
# Open a new Terminal, make sure you are in the (base) conda env
conda env list

# Create F_GPT conda env
conda create -n F_GPT python==3.9 -y

```

* [Recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
```Shell
### TRY Recurrent PPO via sb3-contrib
pip install sb3-contrib

```


```Shell


ipython -c "%run /home/ziangcao2022/workspace/CS330/STF_CS330_FastGPT/src/gpt-new_action.ipynb"



ipython -c "%run /scratch/ziang/round-10/projects/Trojan_Competition/src/Apply_noise/store_titration.ipynb"

export CUDA_VISIBLE_DEVICES=6; ipython -c "%run /scratch/ziang/round-10/projects/Trojan_Competition/src/Apply_noise/store_titration.ipynb"

```



##
* I tried to bypass the env_wrapper and direct integrate with F_GPT or un-structured simulator, the most standard agent RL algorithm will not be that easy to plug in. The challenges are listed below:
	- I believe we still need an env_wrapper, I provide my MetaDrive_Wrapper to help understand.
	- Typically, we can ask RL algo to run as many episodes as we want. However, i do not know how to integral this with we the LLM dataset...
		+ In RL general algo, I just need to call reset() and start next rollouts. Here, we might view the process of generating one full sentence as one episode.
	

```Python
size_vocab = 6000
class FGPT_ENV(gym.Env):
    def __init__(self, render=False):
        self._seed = 
        # This will be the logits 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size_vocab,), dtype=np.float32)
        # Action Space, whether perturb or not

    def step(self, action):
    	# Within the same sentence, process the next word
        return observation, rewards, done, info

    def reset(self):
    	# Start a new sentence
        self.env.reset()
        return self.get_obs()

    def close(self):
    	# Remove the entire FGPT_ENV from GPU
        # print("Closing environment")
        self.env.close()

    def seed(self, seed=None):
            self._seed = np.random.seed(0)

    # Private methods
    def _get_obs(self):
    	# Get the logits ...
        return obs.astype(np.float32)
```





# General Training Agent -- For Single Agent Env
```Python
def _init_agent(args, single_env, e_dr_sim):
    if args.use_sac_agent:
    	# Note: I believe we don't have enough GPU to make multi_env
        agent = SAC("MlpPolicy",    # policy type
                single_env,            # environment
                verbose=1,      # print progressbar
                learning_starts=100,
                gradient_steps=32,
                batch_size=32,
                train_freq=8,
                ent_coef=0.005,
                # action_noise = NormalActionNoise(mean=np.zeros(4), sigma=0.5 * np.ones(4)),
                # policy_kwargs=dict(net_arch=[32, 32]),
                tensorboard_log=args.logdir_train_agent
                )
    else:
        agent = DummyAgent(args, single_env)
    return agent


```