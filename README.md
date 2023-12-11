# STF_CS330_FastGPT

Use RL to predict perturbations to language model's outputs.

## Setup

Use the below to install the F_GPT environment.

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

Then either run `src/fast_detect_gpt/gpt_batch_a1.ipynb`, or run `python3 private_support_code/fast-detect-gpt/GPT_TRUE_BATCH_A1.py` or `python3 private_support_code/fast-detect-gpt/GPT_A1.py`.

```