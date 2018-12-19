# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)

This codebase is based on a forked version of [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl), which implemented Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in Pytorch. This codebase extends the original implementation by adding the option to train with A2C and PPO as well as the original policy methods VPG and TRPO. An additional environment is included called PendulumTheta-v0.

Dependencies are the same as the original codebase.

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with MAML.
```
python main.py --env-name PendulumTheta-v0 --num-workers 8 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 10 --num-layers 2 --hidden-size 100 --num-batches 200 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder output_folder --debug-file debug --device cpu --cliprange 0.2 --noptepochs 2 --nminibatches 4 --ppo_lr 0.001  --useSGD --ppo_momentum 0 --grad_clip 200
```

To train with PPO, use the option --usePPO
The additional PPO parameters are --cliprange --noptepochs --nminibatches --ppo_lr --useSGD --ppo_momentum -- grad_clip
To train with A2C, use the option --baseline 'critic separate'

## References
This project is based on the original implementation of MAML [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in Pytorch. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]
