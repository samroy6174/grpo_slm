# GRPO-SLM

This repository implements **Group Relative Policy Optimization (GRPO)** from scratch in PyTorch to train Small Reasoning LMs, following along with the video tutorial by Avishek Biswas https://www.youtube.com/watch?v=yGkJj_4bjpE 

## Structure

- `grpo_slm/`: core package  
  - `env.py`: a simple gym.Env for reasoning tasks  
  - `model.py`: wraps a Hugging Face causal LM + tokenizer  
  - `grpo.py`: GRPO & PPO loss implementations  
  - `train.py`: end-to-end RL training loop  
- `notebooks/grpo_tutorial.ipynb`: a step-by-step notebook reproducing each code frame  
- `requirements.txt`: required Python packages  
- `setup.py`: package install script

## Installation

```bash
git clone https://github.com/samroy6174/grpo_slm.git
cd grpo_slm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## References

Code from https://www.youtube.com/watch?v=yGkJj_4bjpE
