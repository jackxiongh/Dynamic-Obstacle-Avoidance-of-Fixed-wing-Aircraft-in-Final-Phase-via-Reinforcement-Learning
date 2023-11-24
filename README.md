# Dynamic Obstacle Avoidance of Fixed-wing Aircraft in Final Phase via Reinforcement Learning

# Version 2023.11

# Requirement 
`python 3.8`
`numpy 1.18.5`
`torch 1.10.0`
`pandas 1.3.5`
`gym 0.10.5`



### Introduction

The codes provide a Reinforcement Learning (RL)-based obstacle avoidance strategy for aircraft. The RL-based obstacle avoidance strategy is based on the SAC RL algorithm and includes a pre-training stage and a fine-tuning stage.
The codes also include a baseline 3DVO-based obstacle avoidance strategy.
The codes require an X-plane simulator (version), Pytorch (version), and ... 

### Files

## SAC RL algorithm
# `SAC.py` - Establish an RL-based navigator according to the SAC RL algorithm
# `ReplayBuffer.py` - Establish a replay buffer

## pre-training stage
# `main_pretrain.py` - The main file of pre-training
# `main_pretrain_test.py` - Test a pre-trained RL-based navigator
# `Env_pre.py` - Environment for pre-training and testing a pre-trained RL-based navigator

## fine-tuning stage
# `main.py` - The main file of fine-tuning
# `env.py` - Environment for fine-tuning
# `main_test` - Test a fine-turned RL-based navigator
# `Env_test.py` - Environment for testing a fine-turned RL-based navigator

## 3DVO-based obstacle avoidance strategy
# `main_test_vo` - Test the 3DVO-based obstacle avoidance strategy
# `Env_test_vo.py` - Environment for testing the 3DVO-based obstacle avoidance strategy




