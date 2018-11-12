import maml_rl.envs
import gym
import numpy as np
import torch
import json
import os

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

#changed aggregate to sum from mean for armd bandit problem.
def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

class args:
    def __init__(self):
        self.env_name = 'Bandit-K5-v0'
        self.num_workers = 8
        self.fast_lr = 0.3
        self.max_kl=0.1
        self.fast_batch_size=10   #number of episodes
        self.meta_batch_size = 40 #number of tasks
        self.num_layers = 2
        self.hidden_size = 100
        self.num_batches=2 #100. Number of iterations
        self.output_folder = 'maml-mab-dir'
        self.gamma = 0.99
        self.tau = 1.0
        self.cg_damping = 1e-5
        self.ls_max_step= 15
        self.device = 'cpu'
        self.first_order = False
        self.cg_iters = 10
        self.ls_max_steps = 10
        self.ls_backtrack_ratio = 0.5
args = args()

batch = 800
save_folder = './saves/{0}'.format(args.output_folder)


print(batch)

sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
    num_workers=args.num_workers)

the_model = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(args.hidden_size,) * args.num_layers)

the_model.load_state_dict(torch.load(os.path.join(save_folder,
        'policy-{0}.pt'.format(batch))))

baseline = LinearFeatureBaseline(
    int(np.prod(sampler.envs.observation_space.shape)))

metalearner = MetaLearner(sampler, the_model, baseline, gamma=args.gamma,
    fast_lr=args.fast_lr, tau=args.tau, device=args.device)

test_batch_size = 2
test_reward_before =[]
test_reward_after =[]

for test_batch in range(test_batch_size):
    #sample one task
    test_task = sampler.sample_tasks(num_tasks=1)
    print("test_task: ",test_task)
    sampler.reset_task(test_task[0])

    #sample some episodes for that task
    episodes = metalearner.sample(test_task, first_order=args.first_order)
    test_reward_before.append(total_rewards([ep.rewards for ep, _ in episodes]))
    test_reward_after.append(total_rewards([ep.rewards for _, ep in episodes]))

print("before:",test_reward_before,"; after: ",test_reward_after,"\n")
print("before average: ",np.mean(test_reward_before),
    "after average: ",np.mean(test_reward_after))
